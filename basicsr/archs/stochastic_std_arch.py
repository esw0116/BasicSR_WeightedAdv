import torch
from torch import nn as nn

from .arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class StoStdEstimator(nn.Module):
    """EDSR network structure.
    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=16,
                 rand_std = 1,
                 ):
        super(StoStdEstimator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(num_in_ch*2, num_feat, 3, 1, 1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU())
        self.conv4 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.rand_std = rand_std
        
    def forward(self, sr, gt, exploit=1, get_std=False):
        res = sr - gt
        std = torch.cat((sr, res), dim=1)
        std = self.conv1(std)
        std = self.conv2(std)
        std = self.conv3(std)
        std = self.conv4(std)

        std = std.abs()

        if get_std:
            return std
        
        if self.training:
            if exploit > 1:
                # n = std.new_empty(exploit, *std.shape)
                n = torch.randn((exploit, *std.shape), dtype=std.dtype, layout=std.layout, device=std.device)
                n = n * self.rand_std
                mean = sr.unsqueeze(0)
                std = std.unsqueeze(0)
            else:
                n = torch.randn_like(std)
            sample = mean + n * std
            return sample
        else:
            return sr