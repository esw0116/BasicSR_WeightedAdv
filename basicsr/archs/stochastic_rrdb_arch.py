import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle


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
class Stochastic_RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(Stochastic_RRDBNet, self).__init__()
        self.scale = scale
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

        mean, std = out.chunk(2, dim=1)
        std = std.abs()
        if get_stat:
            return mean, std

        if self.training:
            if exploit > 1:
                n = std.new_empty(exploit, *std.shape)
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
            else:
                n = torch.randn_like(std)
            sample = mean + n * std
            return sample
        else:
            return mean

        return out