# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.common import * 

class TRAN_ST(nn.Module):
    
    def __init__(self, scale: int = 4, n_feats: int = 64, out_channels=1, mean: float = 0., std: float = 0.) :
        super(TRAN_ST, self).__init__()
        self.dem_conv = nn.Conv2d(1, n_feats, kernel_size=3, padding=1)  # DEM
        self.intensity_conv = nn.Conv2d(1, n_feats, kernel_size=3, padding=1)  # INTENSITY
        self.coherence_conv = nn.Conv2d(1, n_feats, kernel_size=3, padding=1)  # COHERENCE
        self.channel_attention = CALayer(n_feats*3)
        self.channel_reduction = nn.Conv2d(n_feats*3, n_feats, kernel_size=1)
        m_body = TRAB(n_feats, n_feats)
        m_tail = [default_conv(n_feats, out_channels, 3)] 
        self.body = nn.Sequential(m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, input: torch.Tensor, feats: list) -> torch.Tensor:
        intensity = feats[0]  
        coherence = feats[1]   
        dem_features = self.dem_conv(input) 
        intensity_features = self.intensity_conv(intensity) 
        coherence_features = self.coherence_conv(coherence) 
        dem_features = F.interpolate(dem_features, size=(192, 192), mode='nearest', align_corners=False) 
        fused_features = torch.cat((dem_features, intensity_features, coherence_features), dim=1)  
        attentioned_features = self.channel_attention(fused_features)                                                                                                            
        reduced_features = self.channel_reduction(attentioned_features)             
        out_fused = self.body(reduced_features) + reduced_features
        output = self.tail(out_fused)
        return output