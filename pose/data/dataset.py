#!/usr/bin/env python3

"""
Dataset modules
"""
import copy
import math
import os

import numpy as np
import orjson
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.io import read_image

import transforms


class CocoDataset(Dataset):

    def __init__(self,
                 anno_fname,
                 image_dir,
                 transform=None,
                 nn_input_size=[256,192],  # height, width
                 do_data_augment=False,
                 flip_proba=0.5,
                 rotate_factor=40,
                 scale_factor=0.3,
                 min_keypoints=1):

        self._init_coco_info()
        with open(anno_fname) as _anno_fin:
            anno = orjson.loads(_anno_fin.read())
        logger.info(f"Annotation fname: {anno_fname}")
        self.image_dir = image_dir
        self.transform = transform
        self.nn_input_size = nn_input_size
        logger.info(f"NN input size: {self.nn_input_size}")
        self.do_data_augment = do_data_augment
        self.flip_proba = flip_proba
        self.rotate_factor = rotate_factor
        self.scale_factor = scale_factor
        logger.info(f"Data augmentation: {self.do_data_augment}")
        # self.half_body_proba = 0.3
        self.half_body_proba = 1.0
        self.half_body_min_kps = 8
        logger.info(f"Half body proba: {self.half_body_proba}, min #kp: {self.half_body_min_kps}")

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

    def _init_coco_info(self):
        self.num_keypoints = 17

        kp_labels = ['nose', 'L-eye', 'R-eye', 'L-ear', 'R-ear',
                     'L-shder', 'R-shder', 'L-elb', 'R-elb',
                     'L-wrst', 'R-wrst', 'L-hip', 'R-hip',
                     'L-knee', 'R-knee', 'L-ankl', 'R-ankl']
        self._keypoints_idx2name = {idx: name for idx, name in enumerate(kp_labels)}
        logger.info(f"Keypoints idx2name: {self._keypoints_idx2name.items()}")

        # using 0-based skeleton index (ref. COCO dataset using 1-based index)
        self.skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
                         [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
                         [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    def __len__(self):
        return len(self._keypoints)

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        image_meta = self.get_image_meta(sample['image_id'])
        image_path = image_meta['file_name']
        image_size = [image_meta['width'], image_meta['height']]
        x, y, bw, bh = sample['bbox']
        center = [int(x+0.5*bw), int(y+0.5*bh)]
        image = read_image(image_path)  # (c,h,w)
        keypoints = sample['keypoints']

        rotate = 0    # angle in degrees (-180 ~ 180)
        scale = 1.25  # default scale
        if self.do_data_augment:
            if np.random.random() < self.half_body_proba \
               and sum(keypoints[:,2] > 0) > self.half_body_min_kps:
                keypoints, bbox, center = self._half_body_transform(keypoints, image_size)
                x, y, bw, bh = bbox
                logger.debug("the image is half-body transformed.")
            if np.random.random() < self.flip_proba:
                image = image.flip(2)
                center[0] = image_size[0] - center[0]
                for i in range(keypoints.shape[0]):
                    if keypoints[i,2] == 0:   # visibility
                        continue
                    keypoints[i,0] = image_size[0] - keypoints[i,0]
                logger.debug("the image is horizontally flipped.")
            rotate_factor = self._get_rotate_factor(image_size[0], center[0])
            rotate = np.clip(np.random.randn() * rotate_factor,
                             rotate_factor * -2, rotate_factor * 2)
            scale = scale * np.clip(np.random.randn() * self.scale_factor + 1,
                                    1 - self.scale_factor, 1 + self.scale_factor)
        logger.debug(f"rotate: {rotate:.2f}, scale: {scale:.2f}")

        image = transforms.rotate_image(image, rotate, center)
        keypoints = transforms.rotate_keypoints(keypoints, rotate,
                                                center, image_size)

        # resizing bbox & cropping
        image, keypoints = transforms.crop_image(image, keypoints, scale, center,
                                                 bw, bh, self.nn_input_size)

        if self.transform is not None:
            image = self.transform(image)

        #TODO: Generating SimCC x,y label.

        return image, keypoints

    def _get_rotate_factor(self, image_width, center_x):
        """
        Retrict rotating factor when side margin is small.
        """
        side_margin = min(center_x, image_width-center_x)
        ratio = float(side_margin) / image_width
        if ratio < 0.1:
            return 5
        elif ratio < 0.2:
            return 10
        else:
            return self.rotate_factor

    def _half_body_transform(self, keypoints, image_size):
        num_upperbody_kps = sum(keypoints[:13,2] > 0) # 13: L-knee
        num_lowerbody_kps = sum(keypoints[11:,2] > 0) # 11: L-hip
        min_kps = 2

        if num_upperbody_kps < min_kps:
            do_upperbody_transf = False
        elif num_lowerbody_kps < min_kps:
            do_upperbody_transf = True
        else:
            if np.random.randn() < 0.5:
                do_upperbody_transf = True
            else:
                do_upperbody_transf = False

        transf_part = "upper" if do_upperbody_transf else "lower"
        logger.debug(f"half body transform part: {transf_part}")

        dst_keypoints = torch.zeros(keypoints.shape, dtype=torch.int32)
        for i in range(keypoints.shape[0]):
            if do_upperbody_transf is True and i >= 13:
                continue
            if do_upperbody_transf is False and i < 11:
                continue
            if keypoints[i,2] == 0:   # visibility
                continue
            dst_keypoints[i] = keypoints[i]

        iw, ih = image_size
        valid_keypoints = dst_keypoints[dst_keypoints[:,2] > 0]
        bw = max(valid_keypoints[:,0]) - min(valid_keypoints[:,0])
        bh = max(valid_keypoints[:,1]) - min(valid_keypoints[:,1])
        center_x = min(valid_keypoints[:,0]) + math.ceil(bw * 0.5)
        center_y = min(valid_keypoints[:,1]) + math.ceil(bh * 0.5)

        scale_factor = 1.5
        bw = bw * scale_factor
        half_bw = min(bw * 0.5, center_x, iw - center_x)
        bh = bh * scale_factor
        half_bh = min(bh * 0.5, center_y, ih - center_y)
        bw = math.ceil(half_bw * 2)
        bh = math.ceil(half_bh * 2)

        x1 = math.ceil(center_x - half_bw)
        y1 = math.ceil(center_y - half_bh)

        bbox = [x1, y1, bw, bh]
        center = [center_x, center_y]

        return dst_keypoints, bbox, center

    def get_connected_keypoints(self, keypoint_idx):
        ret = set()
        for a, b in self.skeleton:
            if a == keypoint_idx:
                ret.add(b)
            elif b == keypoint_idx:
                ret.add(a)
        return list(ret)

    def get_image_meta(self, image_idx):
        return copy.deepcopy(self._image_meta[image_idx])

    def get_sample(self, sample_idx):
        return copy.deepcopy(self._keypoints[sample_idx])

    def keypoints_idx2name(self, keypoint_idx):
        return self._keypoints_idx2name[keypoint_idx]
