import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle

def get_mean(logits: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    '''
    logits: (B, M, 1, H, W)
    means:  (B, M, C, H, W)
    '''

    probs = logits.softmax(dim=1)
    ret = probs * means
    ret = ret.sum(dim=1)
    return ret

def get_std(logits: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    '''
    logits: (B, M, 1, H, W)
    means:  (B, M, C, H, W)
    stds:  (B, M, C, H, W)
    '''
    probs = logits.softmax(dim=1)
    ex2 = probs * (means.pow(2) + stds.pow(2))
    ex2 = ex2.sum(dim=1)

    # (B, C, H, W)
    ex = get_mean(logits, means)
    var = ex2 - ex.pow(2)
    std = (var + 1e-6).sqrt()
    return std

def get_dist(logits: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    dist_map = 0
    n_mods = logits.size(1)
    for idx in range(n_mods - 1):
        if idx == 0:
            dist = means[:, n_mods - 1] - means[:, 0]
        else:
            dist = means[:, idx + 1] - means[:, idx]

        dist_map += dist.pow(2)

    dist_map /= n_mods
    return dist_map

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # num_grow_ch: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.RDB2 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.RDB3 = ResidualDenseBlock_5C(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

@ARCH_REGISTRY.register()
class Stochastic_MMRRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale

        # Check here
        # 1) Can we set them (n_colors, n_mods) as arguments?
        self.n_colors = 3
        self.n_mods = 2
        # {prob + (mean, std) * (R, G, B)} * n_mods
        num_out_ch = (1 + 2 * self.n_colors) * self.n_mods

        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, exploit=1, get_stat=False):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        fea = self.conv_first(feat)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        # Multimodal sampling
        b, _, h, w = out.size()
        out = out.view(b, self.n_mods, -1, h, w)
        logits, means, stds = x.split((1, self.n_colors, self.n_colors), dim=2)
        stds = stds.abs()
        # Check here:
        # 1) Do we need to return sampled SR results? or a single SR result?
        sr = get_mean(logits, means)
        std = get_std(logits, means, stds)
        dist = get_dist(logits, means, stds)

        if get_stat:
            return sr, std, dist
        else:
            return sr