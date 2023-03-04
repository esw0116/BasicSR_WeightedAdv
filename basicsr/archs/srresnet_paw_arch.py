from torch import nn as nn
from torch.nn import functional as F

from utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer


def get_mean(logits, means):
    '''
    logits: (B, M, 1, H, W)
    means:  (B, M, C, H, W)
    '''

    probs = logits.softmax(dim=1)
    ret = probs * means
    ret = ret.sum(dim=1)
    return ret


@ARCH_REGISTRY.register()
class MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        n_mixtures = 2
        # Just ignore std
        num_out_ch_paw = (1 + 2 * num_out_ch) * n_mixtures
        self.conv_last = nn.Conv2d(num_feat, num_out_ch_paw, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        #out += base

        b, _, h, w = out.size()
        out = out.view(b, self.n_mixtures, -1, h, w)
        logits, means, _ = out.split((1, self.num_out_ch, self.num_out_ch), dim=2)
        # Residual connection
        means = means + base.view(base.size(0), 1, base.size(1), base.size(2), base.size(3))
        stds = stds.abs()

        sr = get_mean(logits, means)
        return sr, means, logits


def get_weight(hr, sr, means, logits):
    l_error = (hr - sr).abs()

    l_uncertainty = 0
    probs = logits.softmax(1)
    for i in range(means.size(1)):
        d = (hr - means[:, i]).abs()
        ent = probs[:, i] * (probs[:, i] + 1e-6).log()
        l_uncertainty -= d * ent

    weight = l_error * l_uncertainty
    return weight


from basicsr.utils import resize
def backprojection_loss(means, lr):
    '''
        sr: (B, M, C, H, W)
        lr: (B, C, H, W)
    '''
    b, m, c, h, w = means.shape
    means = means.view(b * m, c, h, w)
    means = means.contiguous()
    means_down = resize.imresize(
        means,
        sizes=(lr.shape[-2], lr.shape[-1]),
        kernel='cubic',
        antialiasing=True,
    )
    means_down = means_down.view(b, m, c, lr.shape[-2], lr.shape[-1])

    lr_repeat = lr.unsqueeze(1).repeat(1, means_down.size(1), 1, 1, 1)
    loss = F.l1_loss(means_down, lr_repeat)
    return loss


def min_ce_loss(gt, means, logits):
    '''
        gt: (B, C, H, W)
        means: (B, M, C, H, W)
    '''
    if means.ndim == gt.ndim:
        raise ValueError('output size should be different from gt size!')

    # (B, 1, C, H, W)
    gts = gt.unsqueeze(1)
    # (B, M, C, H, W)
    gts = gts.repeat(1, means.size(1), 1, 1, 1)

    loss_raw = F.l1_loss(means, gts, reduction='none')

    # Reduce across channel dimension
    # (B, M, H, W)
    loss_raw = loss_raw.mean(dim=2)
    # (B, H, W)
    loss_min, idx_min = loss_raw.min(1)
    loss_min_avg = loss_min.mean()

    logits = logits.squeeze(2)
    # logits:   (B, M, H, W)
    # idx_min:  (B, H, W)
    loss_prob = F.cross_entropy(logits, idx_min)

    w_min = 1
    w_prob = 1
    loss = w_min * loss_min_avg + w_prob * loss_prob
    return loss