import os, glob
import imageio
import torch
from torch import nn
import numpy as np


class WeightMapFT(nn.Module):
    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=16,
                 ):
        super(WeightMapFT, self).__init__()

        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(num_feat)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(num_feat)
        self.conv3 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.sig(4*x)
        return x


ITER = 40000
DATASET = 'B100'

path_net_w = os.path.join('nsml_models/KR80934_CVLAB_SR6_351', str(ITER), 'model/W.pth')

net_w = WeightMapFT()
net_w.load_state_dict(torch.load(path_net_w))

stdmap_path = os.path.join('std_maps', DATASET)
stdmap_files = sorted(glob.glob(os.path.join(stdmap_path, '*.png')))

output_dir = os.path.join('results', 'stdmap_norm_withG', str(ITER))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for stdmap in stdmap_files:
    coeff_name = os.path.basename(stdmap)
    img_coeff = imageio.imread(stdmap)
    img_coeff = img_coeff.astype(np.float16) / 255
    img_coeff = np.sum(img_coeff * (np.array([65.481/255, 128.553/255, 24.966/255]).reshape(1,1,3)), axis=2, keepdims=True) + 16/255

    torch_coeff = torch.from_numpy(img_coeff).permute(2,0,1).unsqueeze(0).float()
    with torch.no_grad():
        output_coeff = net_w(torch_coeff)

    outimg_coeff = output_coeff.squeeze(0).permute(1,2,0).numpy()
    outimg_coeff = (outimg_coeff * 255).round().astype(np.uint8)
    outimg_name = os.path.join(output_dir, coeff_name)
    imageio.imwrite(outimg_name, outimg_coeff)
    # breakpoint()

