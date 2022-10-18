import torch
from torch import nn as nn

from archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from utils.registry import ARCH_REGISTRY

import typing
import math

# from common import default_conv, ResBlock, Upsampler

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size, conv, 
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, conv, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
        

@ARCH_REGISTRY.register()
class EDSRStd(nn.Module):
    '''
    EDSR model

    Note:
        From Lim et al.,
        "Enhanced Deep Residual Networks for Single Image Super-Resolution"
        See https://arxiv.org/pdf/1707.02921.pdf for more detail.
    '''

    def __init__(
            self,
            scale: int=4,
            depth: int=16,
            n_colors: int=3,
            n_feats: int=64,
            res_scale: float=1,
            res_prob: float=1,
            conv=default_conv):

        super().__init__()
        self.n_colors = n_colors
        self.conv = conv(n_colors, n_feats, 3)
        resblock = lambda: ResBlock(
            n_feats, 3, conv=conv, res_scale=res_scale,
        )
        m = [resblock() for _ in range(depth)]
        m.append(conv(n_feats, n_feats, 3))
        self.resblocks = nn.Sequential(*m)
        if scale is None:
            self.recon = None
        elif scale == 1:
            self.recon = conv(n_feats, n_colors, 3)
        else:
            self.recon = nn.Sequential(
                Upsampler(scale, n_feats, conv=conv),
                conv(n_feats, 2 * n_colors, 3),
            )

        self.url = None
        return

    def forward(
            self,
            x: torch.Tensor,
            exploit: int=1,
            get_stat: bool=False) -> torch.Tensor:

        x = self.conv(x)
        x = x + self.resblocks(x)
        if self.recon is None:
            return x

        x = self.recon(x)
        mean, std = x.chunk(2, dim=1)
        std = std.abs()

        return std
        
        # if get_stat:
        #     return mean, std

        # if self.train:
        #     if exploit > 1:
        #         n = std.new_empty(exploit, *std.shape)
        #         mean = mean.unsqueeze(0)
        #         std = std.unsqueeze(0)
        #     else:
        #         n = torch.randn_like(std)

        #     sample = mean + n * std
        #     return sample
        # else:
        #     return mean

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for k in own_state.keys():
            if k not in state_dict and 'body' not in k:
                raise RuntimeError(k + ' does not exist!')
            else:
                if k in state_dict:
                    own_state[k] = state_dict[k]
                elif 'body' in k:
                    k_wob = k.replace('body.', '')
                    own_state[k] = state_dict[k_wob]

        super().load_state_dict(own_state, strict=strict)
