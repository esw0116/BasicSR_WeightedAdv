# This is a pytoch implementation of DISTS metric.
# Requirements: python >= 3.6, pytorch >= 1.0
import argparse

import cv2
import numpy as np
import os, glob
import os.path as osp

import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils import img2tensor


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class DISTS(torch.nn.Module):
    def __init__(self, load_weights=True):
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        if load_weights:
            # weights = torch.load(os.path.join(sys.prefix, 'weights.pt'))
            weights = torch.load('scripts/metrics/dist_weights.pt')
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']
        
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, require_grad=False, batch_average=False):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y) 
        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1+dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score

def prepare_image(image, resize=True):
    if resize and min(image.size) > 256:
        image = transforms.functional.resize(image, 256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)


def main(dataset, network):
    # dataset = args.dataset
    folder_gt = './dataset/benchmark/{}/HR'.format(dataset)
    folder_restored = './results/{}/visualization/{}'.format(network, dataset)

    # Configurations
    # -------------------------------------------------------------------------
    # dataset = 'Urban100'
    # folder_gt = './dataset/benchmark/{}/HR'.format(dataset)
    # folder_restored = './results/ESRGAN_SRx4_Weight_5e-3_sqrt_140000/visualization/{}'.format(dataset)
    # folder_restored = './results/ESRGAN_SRx4_DF2KOST_WeightedGAN/visualization/{}'.format(dataset)
    # folder_restored = './results/ESRGAN_SRx4_DF2KOST_BaseGAN/visualization/{}'.format(dataset)
    # folder_restored = './results/ESRGAN_SRx4_DF2KOST_official/visualization/{}'.format(dataset)
    # crop_border = 4
    suffix = ''
    # -------------------------------------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DISTS().to(device)
    dists_all = []
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

        # calculate dists
        dists_val = model(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())
        dists_val = dists_val.item()
        # breakpoint()

        # print(f'{i+1:3d}: {basename:25}. \tdists: {dists_val:.6f}.')
        dists_all.append(dists_val)

    print(f'\t{dataset}, \t{network}')
    print(f'\tAverage: dists: {sum(dists_all) / len(dists_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', choices=['Set5', 'Set14', 'B100', 'Urban100', 'all'], required=True, help='Dataset type')
    parser.add_argument('--network', '-n', type=str, help='model to test')
    args = parser.parse_args()

    if args.dataset == 'all':
        dataset_list = ['Set5', 'Set14', 'B100', 'Urban100']
        for d in dataset_list:
            main(d, args.network)
    else:
        main(args.dataset, args.network)
