"""
Dataset modules
"""

import copy
import math
import os
from collections.abc import Callable
from typing import Optional, NamedTuple

import numpy as np
import orjson
import torch
import torchvision
from loguru import logger
from torch.utils.data import Dataset
from yacs.config import CfgNode

from pose.data import transforms
from pose.data.datatype import BBox, KeyPoints


class ImageMeta(NamedTuple):
    fpath: str  # image file path
    h: int      # height of the image
    w: int      # width of the image


class DataSample(NamedTuple):
    keypoints: KeyPoints
    bbox: BBox
    image_id: int
    id: int


class CocoDataset(Dataset):
    def __init__(self,
                 anno_fname: str,
                 image_dir: str,
                 nn_input_size: list[int],  # [height, width]
                 data_aug_cfg: CfgNode,
                 min_keypoints: int = 1,
                 transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                 ) -> None:

        self._init_coco_info()
        with open(anno_fname) as _anno_fin:
            anno = orjson.loads(_anno_fin.read())
        logger.info(f"Annotation fname: {anno_fname}")
        self.image_dir = image_dir
        self.nn_input_size = nn_input_size
        logger.info(f"NN input size(height, width): {self.nn_input_size}")
        self.transform = transform

        if data_aug_cfg is None:
            self.do_data_augment = False
            self.flip_proba = 0.0
            self.rotate_factor = 0
            self.scale_factor = 0.0
            self.half_body_proba = 0.0
            self.half_body_min_kps = 100
        else:
            self.do_data_augment = True
            self.flip_proba = data_aug_cfg.FLIP_PROBA
            self.rotate_factor = data_aug_cfg.ROTATE_FACTOR
            self.scale_factor = data_aug_cfg.SCALE_FACTOR
            self.half_body_proba = data_aug_cfg.HALF_BODY_PROBA
            self.half_body_min_kps = data_aug_cfg.HALF_BODY_MIN_KEYPOINTS
        logger.info(f"Data augmentation: {self.do_data_augment}")

        # load image meta data
        self._image_meta = {}
        for _image in anno['images']:
            m = ImageMeta(fpath=os.path.join(self.image_dir, _image['file_name']),
                          h=_image['height'], w=_image['width'])
            self._image_meta[_image['id']] = m
        logger.info(f"Num images: {len(self._image_meta):,d}")

        # load keypoint data
        logger.info(f"Num raw annotations: {len(anno['annotations']):,d}")
        num_skip_crowd = 0
        num_skip_not_human = 0
        num_skip_insufficient_keypoints = 0
        self._dataset = list[DataSample]()
        for _anno in anno['annotations']:
            if _anno['iscrowd'] != 0:
                num_skip_crowd += 1
                continue
            if _anno['category_id'] != 1:  # category_id==1: person
                num_skip_not_human += 1
                continue
            # 3dim(x,y,vis) for 17 keypoints
            kp = KeyPoints(_anno['keypoints'])
            if kp.num_visible() < min_keypoints:
                num_skip_insufficient_keypoints += 1
                continue
            # Convert bbox format from COCO data format(=xywh) to yxhw
            x, y, w, h = [math.floor(_x) for _x in _anno['bbox']]
            bbox = BBox(y=y, x=x, h=h, w=w)
            self._dataset.append(DataSample(keypoints=kp, bbox=bbox,
                                            image_id=_anno['image_id'], id=_anno['id']))
        logger.info(f"skip crowd: {num_skip_crowd:,d}")
        logger.info(f"skip not human: {num_skip_not_human:,d}")
        logger.info(f"skip insufficient keypoints: {num_skip_insufficient_keypoints:,d}")
        logger.info(f"(minimum keypoints cutoff: {min_keypoints})")
        logger.info(f"Num final dataset(keypoints annotation) size: {len(self._dataset):,d}")

    def _init_coco_info(self) -> None:
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

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        # TODO: return type should be specified

        sample = self.get_sample(idx)
        image_meta = self.get_image_meta(sample.image_id)
        image_path = image_meta.fpath
        image_size = [image_meta.w, image_meta.h]
        y, x, bh, bw = sample.bbox.get_yxhw()
        center = [int(x+0.5*bw), int(y+0.5*bh)]
        image = self._read_image(image_path)  # (c,h,w)
        keypoints = sample.keypoints

        rotate = 0    # angle in degrees (-180 ~ 180)
        scale = 1.25  # default scale
        if self.do_data_augment:
            if np.random.random() < self.half_body_proba \
               and keypoints.num_visible() > self.half_body_min_kps:
                keypoints, bbox, center = self._half_body_transform(keypoints, image_size)
                y, x, bh, bw = bbox.get_yxhw()
                logger.debug("the image is half-body transformed.")
            if np.random.random() < self.flip_proba:
                image = image.flip(2)
                center[0] = image_size[0] - center[0]
                keypoints = keypoints.hflip(image_size[0])
                logger.debug("the image is horizontally flipped.")
            rotate_factor = self._get_rotate_factor(image_size[0], center[0])
            rotate = np.clip(np.random.randn() * rotate_factor,
                             rotate_factor * -2, rotate_factor * 2)
            scale = scale * np.clip(np.random.randn() * self.scale_factor + 1,
                                    1 - self.scale_factor, 1 + self.scale_factor)
        logger.debug(f"rotate: {rotate:.2f}, scale: {scale:.2f}")

        image = transforms.rotate_image(image, rotate, center)
        keypoints = keypoints.rotate(rotate, center, image_size)

        # resizing bbox & cropping
        image, bbox = transforms.crop_and_resize_image(image, scale, center,
                                                       bw, bh, self.nn_input_size)
        keypoints = keypoints.stretch(bbox, self.nn_input_size)

        if self.transform is not None:
            nn_input_image = self.transform(image)
        else:
            nn_input_image = copy.deepcopy(image)

        # TODO: Generating SimCC x, y labels.
        # x_labels = None
        # y_labels = None
        # label_weights = None

        # return nn_input_image, x_labels, y_labels, label_weights, image, keypoints
        return nn_input_image, keypoints

    def _get_rotate_factor(self, image_width: int, center_x: int) -> float:
        """Retrict rotating factor when side margin is small."""
        side_margin = min(center_x, image_width - center_x)
        ratio = float(side_margin) / image_width
        if ratio < 0.1:
            return 5
        elif ratio < 0.2:
            return 10
        else:
            return self.rotate_factor

    def _half_body_transform(self,
                             keypoints: KeyPoints,
                             image_size: list[int]
                             ) -> tuple[KeyPoints, BBox, list[int]]:
        min_kps = 2

        if keypoints.num_upperbody_visible() < min_kps:
            do_upperbody_transf = False
        elif keypoints.num_lowerbody_visible() < min_kps:
            do_upperbody_transf = True
        else:
            if np.random.randn() < 0.5:
                do_upperbody_transf = True
            else:
                do_upperbody_transf = False

        transf_part = "upper" if do_upperbody_transf else "lower"
        logger.debug(f"half body transform part: {transf_part}")

        if do_upperbody_transf is True:
            dst_keypoints = keypoints.remove_lowerbody_kps()
        else:
            dst_keypoints = keypoints.remove_upperbody_kps()
        logger.debug(f"visible kps: {dst_keypoints.num_visible()}")

        iw, ih = image_size
        bw, bh, center_x, center_y = dst_keypoints.get_bbox_size_and_center()

        scale_factor = 1.5
        bw = math.ceil(bw * scale_factor)
        half_bw = min(bw * 0.5, center_x, iw - center_x)
        bh = math.ceil(bh * scale_factor)
        half_bh = min(bh * 0.5, center_y, ih - center_y)
        bw = math.ceil(half_bw * 2)
        bh = math.ceil(half_bh * 2)

        x1 = math.ceil(center_x - half_bw)
        y1 = math.ceil(center_y - half_bh)

        bbox = BBox(y=y1, x=x1, h=bh, w=bw)
        center = [center_x, center_y]

        return dst_keypoints, bbox, center

    def _read_image(self, image_path: str) -> torch.Tensor:
        """
        torchvision.io.read_image() wrapper function
        to guarantee number of color channels to 3.
        """
        image = torchvision.io.read_image(image_path)
        if image.shape[0] == 3:
            return image
        elif image.shape[0] == 1:
            # https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/2
            return image.repeat(3, 1, 1)
        else:
            raise ValueError("Invalid #color channel(={image.shape[0]}): {image_path}")

    def get_connected_keypoints(self, keypoint_idx: int) -> list[int]:
        ret = set()
        for a, b in self.skeleton:
            if a == keypoint_idx:
                ret.add(b)
            elif b == keypoint_idx:
                ret.add(a)
        return list(ret)

    def get_image_meta(self, image_idx: int) -> ImageMeta:
        return copy.deepcopy(self._image_meta[image_idx])

    def get_sample(self, sample_idx: int) -> DataSample:
        return copy.deepcopy(self._dataset[sample_idx])

    def keypoints_idx2name(self, keypoint_idx: int) -> str:
        return self._keypoints_idx2name[keypoint_idx]
