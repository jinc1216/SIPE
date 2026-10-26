import os
import cv2
import random
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from .data_util import make_dataset


def random_resize(img, scale_factor=1.0):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired LR/HR dataset for restoration and super-resolution.

    This SR-aware implementation is adapted from ITER and fixes the main
    mismatch in SIPE-SR: LR and HR are no longer required to have the same
    spatial resolution.  ``scale`` must be 1, 2, or 4 and paired_random_crop
    crops HR patches of ``gt_size`` and corresponding LR patches of
    ``gt_size // scale``.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.scale = int(opt.get('scale', 1))
        if self.scale not in (1, 2, 4):
            raise ValueError(f'PairedImageDataset for SIPE-SR expects scale 1/2/4, got {self.scale}.')

        max_num = opt.get('max_num', 999999999)
        self.gt_paths = sorted(make_dataset(self.gt_folder))[:max_num]
        self.lq_paths = sorted(make_dataset(self.lq_folder))[:max_num]

        if opt.get('phase') == 'train':
            filter_key = opt.get('filter_key', None)
            if filter_key:
                self.gt_paths = [p for p in self.gt_paths if filter_key not in p]
                self.lq_paths = [p for p in self.lq_paths if filter_key not in p]

        if len(self.gt_paths) != len(self.lq_paths):
            raise ValueError(f'GT/LQ file count mismatch: {len(self.gt_paths)} vs {len(self.lq_paths)}. ')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.gt_paths[index]
        lq_path = self.lq_paths[index]
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)
        if img_gt is None:
            raise FileNotFoundError(f'Failed to read GT image: {gt_path}')
        if img_lq is None:
            raise FileNotFoundError(f'Failed to read LQ image: {lq_path}')
        img_gt = img_gt.astype(np.float32) / 255.0
        img_lq = img_lq.astype(np.float32) / 255.0

        h_gt, w_gt = img_gt.shape[:2]
        h_lq, w_lq = img_lq.shape[:2]
        if h_gt != h_lq * self.scale or w_gt != w_lq * self.scale:
            raise ValueError(
                f'Scale mismatch for {os.path.basename(gt_path)} / {os.path.basename(lq_path)}: '
                f'GT {(h_gt, w_gt)} is not {self.scale}x LQ {(h_lq, w_lq)}.')

        if self.opt['phase'] == 'train':
            gt_size = int(self.opt.get('gt_size', h_gt))
            if gt_size % self.scale != 0:
                raise ValueError(f'gt_size={gt_size} must be divisible by scale={self.scale}.')
            if self.opt.get('use_resize_crop', False):
                min_gt = min(h_gt, w_gt)
                if min_gt > gt_size:
                    random_gt_size = random.randint(gt_size, min_gt)
                    random_gt_size = random_gt_size - random_gt_size % self.scale
                    resize_factor = random_gt_size / min_gt
                    img_gt = random_resize(img_gt, resize_factor)
                    img_lq = random_resize(img_lq, resize_factor)
            if img_gt.shape[0] > gt_size or img_gt.shape[1] > gt_size:
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, self.scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt.get('use_flip', False), self.opt.get('use_rot', False))
        else:
            crop_eval_size = self.opt.get('crop_eval_size', None)
            if crop_eval_size:
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, int(crop_eval_size), self.scale, gt_path)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.gt_paths)
