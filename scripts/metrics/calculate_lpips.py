import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def main():
    # Configurations
    # -------------------------------------------------------------------------
    dataset = 'Urban100'
    folder_gt = './dataset/benchmark/{}/HR'.format(dataset)
    folder_restored = './results/ESRGAN_SRx4_DF2KOST_BaseGAN/visualization/{}'.format(dataset)
    # folder_restored = './results/ESRGAN_SRx4_DF2KOST_WeightedGAN/visualization/{}'.format(dataset)
    # crop_border = 4
    suffix = ''
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_gt_list = sorted(glob.glob(osp.join(folder_gt, '*')))
    img_sr_list = sorted(glob.glob(osp.join(folder_restored, '*')))
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, (img_gt_path, img_sr_path) in enumerate(zip(img_gt_list, img_sr_list)):
        basename, ext = osp.splitext(osp.basename(img_gt_path))
        img_gt = cv2.imread(img_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(img_sr_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # img_restored = cv2.imread(osp.join(folder_restored, basename + suffix + ext), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        
        if img_gt.ndim == 2:
            img_gt = np.stack([img_gt, img_gt, img_gt], axis=-1)
            img_restored = np.stack([img_restored, img_restored, img_restored], axis=-1)
        
        h, w = img_restored.shape[:2]
        img_gt = img_gt[:h, :w]
        
        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())
        lpips_val = lpips_val.item()
        # breakpoint()

        print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val:.6f}.')
        lpips_all.append(lpips_val)

    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':
    main()
