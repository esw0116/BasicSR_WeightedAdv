import torch
from torch import nn

from archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class WeightMapNonNorm3FT(nn.Module):
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
                 ndim_norm=1
                 ):
        super(WeightMapNonNorm3FT, self).__init__()

        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.sig = nn.Sigmoid()

        self.ndim_norm = ndim_norm

    def norm(self, w):
        if self.ndim_norm == 1:
            w = w / (w.norm(p=2, dim=1, keepdim=True) + 1e-6)
        elif self.ndim_norm == 3:
            w = w / (w.norm(p=2, dim=(1,2,3), keepdim=True) + 1e-6)
        else:
            w = w / (w.norm(p=2, dim=(0,1,2,3), keepdim=True) + 1e-6)
        return w

    def sum(self, w):
        height, width = w.shape[2:]
        w = 0.25 * height * width * w / (w.sum(dim=(2,3), keepdim=True) + 1e-6)
        w = w.clip(0, 1)
        return w

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.conv3(x)
        # x = self.norm(x)
        x = self.sum(x)

        return x
