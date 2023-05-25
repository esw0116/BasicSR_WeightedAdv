import os
import pickle
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize, to_grayscale
from nsml import DATASET_PATH

from data.data_util import pyweighted_paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from data.weighted_transforms import pyaugment, paired_random_crop
from utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class HybridWeightPairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(HybridWeightPairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.normalize_map = opt['normalize'] if 'normalize' in opt else False
        
        if '+' in opt['dataroot_gt']:
            opt['dataroot_gt'] = opt['dataroot_gt'].split('+')
            opt['dataroot_lq'] = opt['dataroot_lq'].split('+')
            opt['dataroot_weight'] = opt['dataroot_weight'].split('+')

        self.gt_folder = [os.path.join(DATASET_PATH, path) for path in opt['dataroot_gt']]
        self.lq_folder = [os.path.join(DATASET_PATH, path) for path in opt['dataroot_lq']]
        self.weight_folder = [os.path.join(DATASET_PATH, path) for path in opt['dataroot_weight']]

        if 'filename_tmpl' in opt:
            if '+' in opt['filename_tmpl']:
                self.filename_tmpl = opt['filename_tmpl'].split('+')
            elif isinstance(opt['dataroot_gt'], list):
                # self.filename_tmpl = [opt['filename_tmpl']] * len(opt['dataroot_gt'])
                self.filename_tmpl = [opt['filename_tmpl'], '{}']
            else:
                self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if 'filename_tmpl_weight' in opt:
            self.filename_tmpl_weight = opt['filename_tmpl_weight']
        else:
            self.filename_tmpl_weight = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = []
            for i in range(len(self.gt_folder)):
                self.paths.extend(pyweighted_paired_paths_from_folder([self.lq_folder[i], self.gt_folder[i], self.weight_folder[i]], 
                                                             ['lq', 'gt', 'weight'], self.filename_tmpl[i], self.filename_tmpl_weight))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq ], bgr2rgb=True, float32=True)
        # img_gt, img_lq = img_gt / 255, img_lq / 255

        weight_path = self.paths[index]['weight_path']
        img_coeff = torch.load(weight_path)[0]
        if img_coeff.min() < 0:
            img_coeff = img_coeff + 1 # Add 1 for non-negative weight

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, img_coeff = paired_random_crop(img_gt, img_lq, img_coeff, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq, img_coeff = pyaugment([img_gt, img_lq, img_coeff], self.opt['use_hflip'], self.opt['use_rot'])

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {'lq': img_lq, 'gt': img_gt, 'coeff': img_coeff, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)