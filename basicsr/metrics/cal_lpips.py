import numpy as np
import torch
import torch.nn.functional as F
import lpips
from torchvision.transforms.functional import normalize

from utils.registry import METRIC_REGISTRY
from utils import img2tensor


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    img, img2 = img2tensor([img, img2], bgr2rgb=True, float32=True)
    img, img2 = img/255, img2/255
    # print('Before normalization', img.max(), img.min(), img2.max(), img2.min())
    # norm to [-1, 1]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize(img, mean, std, inplace=True)
    normalize(img2, mean, std, inplace=True)
    # print('After normalization', img.max(), img.min(), img2.max(), img2.min())

    loss_fn_vgg = lpips.LPIPS(net='alex', verbose=False).cuda()  # RGB, normalized to [-1,1]
    # calculate lpips
    lpips_val = loss_fn_vgg(img.unsqueeze(0).cuda(), img2.unsqueeze(0).cuda())
    lpips_val = lpips_val.item()

    return lpips_val

