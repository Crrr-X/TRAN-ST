# -*- encoding: utf-8 -*-

from __future__ import annotations
import torch
import torch.nn as nn
from src.loss.aspect_loss import AspectLossFunc as AspectLoss
from src.loss.slope_loss import SlopeLossFunc as SlopeLoss


class Loss(object):
    """init loss module
    """

    def __init__(self, weight) -> None:
        self.weight = weight
    
    def l1_loss(self, sr: torch.Tensor, hr: torch.Tensor, feats = [], epoch: int = -1) -> torch.Tensor:
        mask = ~torch.isnan(hr)
        criterion = nn.L1Loss()
        temp_hr = torch.where(mask, hr, sr) 
        intensity, coherence = torch.zeros((16, 1, 192, 192)).to(sr.device), torch.zeros((16, 1, 192, 192)).to(sr.device) # 初始化特征张量
        if len(feats):
            intensity = feats[0]
            coherence = feats[1]
        loss = criterion(sr[mask], hr[mask]) + criterion(sr[mask] * intensity[mask], hr[mask] * intensity[mask]) + criterion(sr[mask] * coherence[mask], hr[mask] * coherence[mask]) + self.SlopeLoss(sr, temp_hr, 3)+self.AspectLoss(sr, temp_hr, 3)
        
        return loss

    def l2_loss(self, x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        criterion = nn.MSELoss(reduction=reduction)
        loss = criterion(x, y)
        return loss

    def SlopeLoss(self, sr, hr, resolution):
        criterion = SlopeLoss(epsilon=1e-8, resolution=resolution)
        loss = criterion(sr, hr)
        return loss
    
    def AspectLoss(self, sr, hr, resolution):
        criterion = AspectLoss(epsilon=1e-8, resolution=resolution)
        loss = criterion(sr, hr)
        return loss