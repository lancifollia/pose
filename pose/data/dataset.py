#!/usr/bin/env python3

"""
Dataset modules
"""
import os

import orjson as json
import torch
from loguru import logger
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, anno_fname, image_dir):
        self._init_coco_info()

        self._image_dir = image_dir
        with open(anno_fname) as _anno_fin:
            anno = json.loads(_anno_fin.read())
        logger.info(f"Annotation fname: {anno_fname}")

        # load image meta data
        self._image_meta = {}
        for _img in anno['images']:
            v = {}
            for att in ['file_name', 'height', 'width']:
                if att == 'file_name':
                    v[att] = os.path.join(self._image_dir, _img[att])
                else:
                    v[att] = _img[att]
            self._image_meta[_img['id']] = v
        logger.info(f"Num images: {len(self._image_meta):,d}")

        # load keypoint data
        self.keypoints = []
        for _anno in anno['annotations']:
            if _anno['iscrowd'] != 0 or _anno['category_id'] != 1:  # category_id==1: person
                continue
            v = {'keypoints': torch.tensor(_anno['keypoints'], dtype=torch.int32).reshape(17,3), # 17x3(x,y,visibility) matrix
                 'bbox': [round(_x) for _x in _anno['bbox']], # [leftupper_x, leftupper_y, width, height]
                 'image_id': _anno['image_id'],
                 'id': _anno['id']}
            self.keypoints.append(v)
        logger.info(f"Num keypoint data samples: {len(self.keypoints):,d}")

    def _init_coco_info(self):
        self.num_keypoints = 17

        keypoints = ['nose', 'L-eye', 'R-eye', 'L-ear', 'R_ear',
                     'L-shder', 'R-shder', 'L-elb', 'R-elb',
                     'L-wrst', 'R-wrst', 'L-hip', 'R-hip',
                     'L-knee', 'R-knee', 'L-ankl', 'R-ankle']
        self._keypoints_idx2name = {idx: name for idx, name in enumerate(keypoints)}
        logger.info(f"Keypoints idx2name: {self._keypoints_idx2name.items()}")

        self._skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                          [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                          [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return self.keypoints[idx]

    def get_image_meta(self, image_idx):
        return self._image_meta[image_idx]

    def get_skeleton(self):
        return self._skeleton

    def keypoints_idx2name(self, idx):
        return self._keypoints_idx2name[idx]
