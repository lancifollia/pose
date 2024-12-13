#!/usr/bin/env python3

"""
Dataset modules
"""
import os

import orjson as json
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.io import read_image


class CocoDataset(Dataset):
    def __init__(self, anno_fname, image_dir, transform=None, target_transform=None, min_keypoints=1):
        self.init_coco_info()

        self.transform = transform
        self.target_transform = target_transform

        self.image_dir = image_dir
        with open(anno_fname) as _anno_fin:
            anno = json.loads(_anno_fin.read())
        logger.info(f"Annotation fname: {anno_fname}")

        # load image meta data
        self._image_meta = {}
        for _image in anno['images']:
            v = {}
            for att in ['file_name', 'height', 'width']:
                if att == 'file_name':
                    v[att] = os.path.join(self.image_dir, _image[att])
                else:
                    v[att] = _image[att]
            self._image_meta[_image['id']] = v
        logger.info(f"Num images: {len(self._image_meta):,d}")

        # load keypoint data
        logger.info(f"Num raw annotations: {len(anno['annotations']):,d}")
        num_skip_crowd = 0
        num_skip_not_human = 0
        num_skip_insufficient_keypoints = 0
        self._keypoints = []
        for _anno in anno['annotations']:
            if _anno['iscrowd'] != 0:
                num_skip_crowd += 1
                continue
            if _anno['category_id'] != 1:  # category_id==1: person
                num_skip_not_human += 1
                continue
            # 3dim(x,y,vis) for 17 keypoints
            kp = torch.tensor(_anno['keypoints'], dtype=torch.int32).reshape(17,3)
            if torch.sum(kp[:,2] > 0) < min_keypoints:
                num_skip_insufficient_keypoints += 1
                continue
            bbox = [round(_x) for _x in _anno['bbox']]  # [leftupper_x, leftupper_y, width, height]
            v = {'keypoints': kp,
                 'bbox': bbox,
                 'image_id': _anno['image_id'],
                 'id': _anno['id']}
            self._keypoints.append(v)
        logger.info(f"skip crowd: {num_skip_crowd:,d}")
        logger.info(f"skip not human: {num_skip_not_human:,d}")
        logger.info(f"skip insufficient keypoints: {num_skip_insufficient_keypoints:,d}")
        logger.info(f"(minimum keypoints cutoff: {min_keypoints})")
        logger.info(f"Num final keypoints annotation samples: {len(self._keypoints):,d}")

    def init_coco_info(self):
        self.num_keypoints = 17

        keypoints = ['nose', 'L-eye', 'R-eye', 'L-ear', 'R-ear',
                     'L-shder', 'R-shder', 'L-elb', 'R-elb',
                     'L-wrst', 'R-wrst', 'L-hip', 'R-hip',
                     'L-knee', 'R-knee', 'L-ankl', 'R-ankle']
        self._keypoints_idx2name = {idx: name for idx, name in enumerate(keypoints)}
        logger.info(f"Keypoints idx2name: {self._keypoints_idx2name.items()}")

        # using 0-based skeleton index (ref. COCO dataset using 1-based index)
        self._skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
                          [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
                          [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    def __len__(self):
        return len(self._keypoints)

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        image_path = self.get_image_meta(sample['image_id'])['file_name']

        image = read_image(image_path)
        if self.transform is not None:
            image = self.transform(image)

        annotation = sample['keypoints']
        if self.target_transform is not None:
            annotation = self.target_transform(annotation)

        return image, annotation

    def get_connected_keypoints(self, keypoint_idx):
        ret = set()
        for a, b in self._skeleton:
            if a == keypoint_idx:
                ret.add(b)
            elif b == keypoint_idx:
                ret.add(a)
        return list(ret)

    def get_image_meta(self, image_idx):
        return self._image_meta[image_idx]

    def get_sample(self, sample_idx):
        return self._keypoints[sample_idx]

    def keypoints_idx2name(self, keypoint_idx):
        return self._keypoints_idx2name[keypoint_idx]
