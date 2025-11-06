import torch
import torch.nn as nn
import math
# import torch.nn.functional as func
# import numpy as np


class AspectLossFunc(nn.Module):
    def __init__(self, epsilon, resolution = 3):
        super(AspectLossFunc, self).__init__()
        self.eps = epsilon
        self.scale = resolution


    def forward(self, sr, hr):
        sr_offset_x = sr[:, :, :, 2:]
        hr_offset_x = hr[:, :, :, 2:]
        sr_offset_y = sr[:, :, 2:, :]
        hr_offset_y = hr[:, :, 2:, :]
        hr_diff_x = (hr[:, :, :, :-2] - hr_offset_x)[:,:,:-2,:] / self.scale
        sr_diff_x = (sr[:, :, :, :-2] - sr_offset_x)[:,:,:-2,:] / self.scale
        hr_diff_y = (hr[:, :, :-2, :] - hr_offset_y)[:,:,:,:-2] / self.scale
        sr_diff_y = (sr[:, :, :-2, :] - sr_offset_y)[:,:,:,:-2] / self.scale
        assert hr_diff_x.size() == hr_diff_y.size()
        assert sr_diff_x.size() == sr_diff_y.size()
        hr_aspect = self.__cacAspect(hr_diff_x, hr_diff_y, self.eps)
        sr_aspect = self.__cacAspect(sr_diff_x, sr_diff_y, self.eps)
        loss = torch.mean(torch.abs(hr_aspect - sr_aspect))
        return loss


    def __cacAspect(self, dx, dy, eps):
        aspect =(torch.atan2(dy, dx + eps)) % 360            
        return aspect
