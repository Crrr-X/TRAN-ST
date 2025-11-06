import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional

def default_conv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias: bool = True) -> nn.Module:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ESA(nn.Module): 
    def __init__(self, channel: int, reduction: int = 4) -> None:
        super().__init__()
        act = nn.ReLU(True) 
        self.channel_reduction = nn.Sequential(
            default_conv(channel, channel//reduction, 1),
            act,
        ) 
        conv_group = [default_conv(channel//reduction, channel//reduction, 3), act, 
            default_conv(channel//reduction, channel//reduction, 3), act,
            default_conv(channel//reduction, channel//reduction, 3), act,]
        self.conv_stride_pool = nn.Sequential(nn.Conv2d(channel//reduction, channel//reduction, 3, dilation=6, padding=6, bias=True),
            act,
            *conv_group)
        self.conv_rd = nn.Sequential(default_conv(2 * channel//reduction, channel, 1),
            nn.Sigmoid())

    def forward(self, x):
        feature = self.channel_reduction(x)
        feature_conv = self.conv_stride_pool(feature)
        feature_sa = self.conv_rd(torch.cat([feature, feature_conv], dim=1)) 
        return feature_sa*x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size 
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)) 

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0]) 
        coords_w = torch.arange(self.window_size[1]) 
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 
        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop) 
        self.proj = nn.Linear(dim, dim) 
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02) 
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, x, mask=None):
        B_, N, C = x.shape 
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale 
        attn = (q @ k.transpose(-2, -1).contiguous()) 

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(  # type: ignore
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) 
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # (nH, Wh*Ww, Wh*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0] 
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0) 
            attn = attn.view(-1, self.num_heads, N, N) 
            attn = self.softmax(attn) 
        else:
            attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C) 
        x = self.proj(x) 
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: int, num_heads: int, window_size: int = 8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, fused_window_process=False, ca_layer = None):
        super().__init__()
        self.dim = dim
        self.resolution = to_2tuple(input_resolution)
        self.nlocal_featuresum_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size" 

        self.norm1 = norm_layer(dim) 
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask) 
        self.fused_window_process = fused_window_process 

        self.ca_layer = ca_layer

    def calculate_mask(self, x_size): 
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt 
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size) 
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape 
        assert L == H * W, "input feature has wrong size"  # type: ignore

        shortcut = x
        x = self.norm1(x) 
        x = x.view(B, H, W, C)
        ## MSA/w-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) 
        else:
            shifted_x = x
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
            # DIFF: self.fused_window_process
        else:
            x = shifted_x

        x = x.view(B, H * W, C)  # type: ignore
        x = shortcut + x 
        x = self.mlp(self.norm2(x)) 

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim: int, input_resolution, embed_dim=50, depth=2, num_heads=8, window_size=8,
                 mlp_ratio=1., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=None, fused_window_process=False, ca_layer: Optional[nn.Module] = None):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.depth = depth
        self.window_size = window_size
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                          num_heads=num_heads, window_size=window_size,
                                                          shift_size=0 if (
                                                              i % 2 == 0) else window_size // 2,
                                                          mlp_ratio=mlp_ratio,
                                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                          drop=drop, attn_drop=attn_drop,
                                                          drop_path=drop_path[i] if isinstance(
                                                              drop_path, list) else drop_path,
                                                          norm_layer=norm_layer,  # type: ignore
                                                          fused_window_process=fused_window_process,
                                                          ca_layer = ca_layer)
                                     for i in range(depth)])
        self.patch_embed = PatchEmbed(embed_dim=dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def check_image_size(self, x):
        _, _, h, w = x.size() 
        mod_pad_h = (self.window_size - h %
                     self.window_size) % self.window_size 
        mod_pad_w = (self.window_size - w %
                     self.window_size) % self.window_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect') 
        return x, h, w

    def forward(self, x):
        x, h, w = self.check_image_size(x)
        _, _, H, W = x.size()
        x_size = (H, W)
        x = self.patch_embed(x) 
        for blk in self.blocks:
            x = blk(x, x_size)
        x = self.patch_unembed(x, x_size) 
        if h != H or w != W: 
            x = x[:, :, 0:h, 0:w].contiguous()
        return x

class SwinTransformer(nn.Module):
    def __init__(
            self, depth: int = 2, n_feats: int = 50, window_size: int = 8, num_heads: int = 8, resolution: int = 48,
            # bias=True, bn=False, act=nn.ReLU(True)):
            mlp_ratio:float = 2.0, ca_layer: Optional[nn.Module] = None):
        super(SwinTransformer, self).__init__()
        m = []
        m.append(BasicLayer(dim=n_feats,
                            depth=depth,
                            input_resolution=resolution,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True, qk_scale=None,
                            norm_layer=nn.LayerNorm, ca_layer=ca_layer))
        self.transformer_body = nn.Sequential(*m)

    def forward(self, x):
        res = self.transformer_body(x)
        return res

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim=50, norm_layer=None):
        super().__init__()
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()  
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, self.embed_dim, x_size[0], x_size[1]) 
        return x


def window_partition(x, window_size): 
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class TRAB(nn.Module):
    def __init__(self, in_channels: int, n_feats: int, depth: int = 2, window_size: int = 8, num_modules: int = 4) -> None:
        super().__init__()
        m_core_list = [self.create_ctb(
            in_channels, n_feats, depth, window_size) for _ in range(num_modules)] 
        self.fusion = nn.Sequential(default_conv(n_feats * num_modules, n_feats, 1))
        self.core = nn.ModuleList(m_core_list)
        self.num_modules = num_modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = []
        for i in range(self.num_modules):
            local_features.append(x)
        out_fused = self.fusion(torch.cat(local_features, 1)) 
        return out_fused

    def create_ctb(self, in_channels: int, n_feats: int, depth: int, window_size: int):
         
        m_core_list = [
            ESA(in_channels),
            RFA(n_feats),
            SwinTransformer(n_feats=n_feats,
                            depth=depth,
                            window_size=window_size),
            default_conv(n_feats, n_feats, 3),
            ESA(in_channels)
        ]
        return nn.Sequential(*m_core_list)

class RFA(nn.Module):
    def __init__(self, n_feats) -> None:
        super(RFA, self).__init__() 
        self.n_resblocks = 4
        rfa = [Res(n_feats=n_feats)
               for _ in range(self.n_resblocks)]
        self.RFA = nn.ModuleList(rfa)
        self.tail = nn.Sequential(
            default_conv(n_feats * 4, n_feats, 1)
        )

    def forward(self, x):
        feature = x
        res_feature = []
        for index in range(self.n_resblocks):
            tmp, x = self.RFA[index](x)
            res_feature.append(tmp)
        res = self.tail(torch.cat(res_feature, dim=1))
        res += feature
        return res

class Res(nn.Module):
    def __init__(self, n_feats, ) -> None:
        super().__init__()
        act = nn.ReLU(True)
        self.body = nn.Sequential(
            default_conv(n_feats, n_feats, 3, bias=True),
            act,
            default_conv(n_feats, n_feats, 3, bias=True),
            act,
            default_conv(n_feats, n_feats, 3, bias=True),
            ESA(n_feats)
        ) 
        
    def forward(self, x):
        res = self.body(x)
        return res, res+x