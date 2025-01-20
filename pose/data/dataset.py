"""
Dataset modules
"""

import copy
import math
import os
from collections import OrderedDict
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Optional, NamedTuple

import numpy as np
import orjson
import torch
import torchvision
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
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
                 cfg: CfgNode,
                 is_train: bool,
                 transform: Optional[Callable[[Tensor], Tensor]] = None
                 ) -> None:
        self._init_coco_info()
        data_cfg = cfg.DATA_TRAIN if is_train else cfg.DATA_VAL

        self.anno_fname = data_cfg.ANNOTATION_FNAME
        with open(self.anno_fname) as _anno_fin:
            anno = orjson.loads(_anno_fin.read())
        logger.info(f"Annotation fname: {self.anno_fname}")
        self.image_dir = data_cfg.IMAGE_DIR
        logger.info(f"Image file director: {self.image_dir}")

        self.nn_input_size = cfg.MODEL.INPUT_SIZE
        logger.info(f"NN input size(height, width): {self.nn_input_size}")
        self.min_keypoints=cfg.DATA_COMMON.MIN_KEYPOINTS

        self.simcc_split_factor = cfg.MODEL.EXTRA.SIMCC_SPLIT_FACTOR
        self.simcc_sigma = cfg.MODEL.EXTRA.SIMCC_SIGMA

        self.transform = transform

        data_aug_cfg = data_cfg.DATA_AUG
        if data_aug_cfg is None:
            self.do_data_augment = False
            self.flip_proba = 0.0
            self.rotate_factor = 0
            self.scale_factor = 0.0
            self.half_body_proba = 0.0
            self.half_body_min_kps = 1e+10
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
        num_skip_invalid_bbox = 0
        self._dataset = list[DataSample]()
        for _anno in anno['annotations']:
            if _anno['iscrowd'] != 0:
                num_skip_crowd += 1
                continue
            if _anno['category_id'] != 1:  # category_id==1: person
                num_skip_not_human += 1
                continue
            # 17 keypoints (COCO dataset format: 3dim(x,y,vis) for each kp)
            kp = KeyPoints(anno=_anno['keypoints'], data_format=self.data_format)
            if kp.num_visible() < self.min_keypoints:
                num_skip_insufficient_keypoints += 1
                continue
            # Sanitize bbox data
            image_meta = self._image_meta[_anno['image_id']]
            ih = image_meta.h
            iw = image_meta.w
            x, y, bw, bh = [math.floor(_x) for _x in _anno['bbox']]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(iw - 1, x1 + max(0, bw - 1))
            y2 = min(ih - 1, y1 + max(0, bh - 1))
            if _anno['area'] > 0 and x2 >= x1 and y2 >= y1:
                # Convert bbox format from COCO data format(=xywh) to yxhw
                bbox = BBox(y=y1, x=x1, h=y2-y1, w=x2-x1)
                self._dataset.append(DataSample(keypoints=kp, bbox=bbox,
                                                image_id=_anno['image_id'], id=_anno['id']))
            else:
                num_skip_invalid_bbox += 1
        logger.info(f"skip crowd: {num_skip_crowd:,d}")
        logger.info(f"skip not human: {num_skip_not_human:,d}")
        logger.info(f"skip insufficient keypoints: {num_skip_insufficient_keypoints:,d}")
        logger.info(f"skip invalid bbox: {num_skip_invalid_bbox:,d}")
        logger.info(f"minimum keypoints cutoff: {self.min_keypoints}")
        logger.info(f"Num final dataset(keypoints annotation) size: {len(self._dataset):,d}")

    def _init_coco_info(self) -> None:
        self.data_format = "COCO"
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

    def __getitem__(self, idx: int)\
        -> tuple[Tensor, Tensor, Tensor, Tensor, Dict]:
        sample = self.get_sample(idx)
        image_meta = self.get_image_meta(sample.image_id)
        image_path = image_meta.fpath
        image_size = [image_meta.h, image_meta.w]
        y, x, bh, bw = sample.bbox.get_yxhw()
        center = [math.floor(y+0.5*bh), math.floor(x+0.5*bw)]
        image = self._read_image(image_path)  # (c,h,w)
        keypoints = sample.keypoints

        rotate = 0    # angle in degrees (-180 ~ 180)
        scale = 1.25  # default scale
        if self.do_data_augment:
            if np.random.random() < self.half_body_proba \
               and keypoints.num_upperbody_visible() > self.half_body_min_kps:
                keypoints, bbox, center = self._half_body_transform(keypoints, image_size)
                _, _, bh, bw = bbox.get_yxhw()
                logger.debug("the image is half-body transformed.")
            if np.random.random() < self.flip_proba:
                image = image.flip(2)
                center[1] = image_size[1] - center[1]
                keypoints = keypoints.hflip(image_size[1])
                logger.debug("the image is horizontally flipped.")
            rotate_factor = self._get_rotate_factor(image_size[1], center[1])
            rotate = np.clip(np.random.randn() * rotate_factor,
                             rotate_factor * -2, rotate_factor * 2)
            scale = scale * np.clip(np.random.randn() * self.scale_factor + 1,
                                    1 - self.scale_factor, 1 + self.scale_factor)
        logger.debug(f"rotate: {rotate:.2f}, scale: {scale:.2f}")

        image = transforms.rotate_image(image, rotate, center)
        keypoints = keypoints.rotate(rotate, center, image_size)

        image, bbox = transforms.crop_and_resize_image(image, scale, center,
                                                       [bh, bw], self.nn_input_size)
        keypoints = keypoints.stretch(bbox, self.nn_input_size)

        if self.transform is not None:
            nn_input_image = self.transform(image)
        else:
            nn_input_image = copy.deepcopy(image)

        target_y, target_x, target_weight = self._generate_simcc_label(keypoints)
        meta = {"image": image,
                "image_id": torch.tensor(sample.image_id),  # image_id: int type
                "bbox": torch.tensor(bbox.get_yxhw()),
                "keypoints": keypoints.get_kps(),
                "vis": keypoints.get_vis()}

        return nn_input_image, target_y, target_x, target_weight, meta

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
        logger.debug("half(upper) body transform")

        dst_keypoints = keypoints.remove_lowerbody_kps()
        logger.debug(f"visible kps: {dst_keypoints.num_visible()}")

        iw, ih = image_size
        bh, bw, center_y, center_x = dst_keypoints.get_bbox_size_and_center()

        scale_factor = 1.5
        bh = math.ceil(bh * scale_factor)
        half_bh = min(bh * 0.5, center_y, ih - center_y)
        bw = math.ceil(bw * scale_factor)
        half_bw = min(bw * 0.5, center_x, iw - center_x)
        bh = math.ceil(half_bh * 2)
        bw = math.ceil(half_bw * 2)

        y1 = math.ceil(center_y - half_bh)
        x1 = math.ceil(center_x - half_bw)

        bbox = BBox(y=y1, x=x1, h=bh, w=bw)
        center = [center_y, center_x]

        return dst_keypoints, bbox, center

    def _read_image(self, image_path: str) -> Tensor:
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
            msg = "_read_image() expected image channel size to be one of 3, or 1 " \
                  f"but not found {image.shape[0]}. image file path: {image_path}"
            raise RuntimeError(msg)


    def _generate_simcc_label(self, keypoints: KeyPoints)\
        -> tuple[Tensor, Tensor, Tensor]:
        target_weight = torch.zeros((self.num_keypoints, 1), dtype=torch.float32)
        target_weight[keypoints.get_vis() > 0] = 1.0

        h, w = self.nn_input_size
        sf = self.simcc_split_factor

        target_y = torch.zeros((self.num_keypoints, h * sf), dtype=torch.float32)
        target_x = torch.zeros((self.num_keypoints, w * sf), dtype=torch.float32)

        sigma = self.simcc_sigma
        buffer = sigma * 3
        for kp_idx in range(self.num_keypoints):
            ky, kx = keypoints.get_kp(kp_idx)

            # Check that any part of the gaussian is in-bounds
            uly, ulx = math.floor(ky - buffer), math.floor(kx - buffer)
            bry, brx = math.ceil(ky + buffer + 1), math.ceil(kx + buffer + 1)
            if uly < 0 or ulx < 0 or bry > h or brx > w:
                target_weight[kp_idx] = 0.0

            if target_weight[kp_idx] < 1e-10:
                continue

            # Space-aware Gaussian label smoothing
            mu_y = ky * sf + math.ceil(sf / 2) - 1
            mu_x = kx * sf + math.ceil(sf / 2) - 1
            y = torch.arange(0, math.ceil(h * sf), 1, dtype=torch.float32)
            x = torch.arange(0, math.ceil(w * sf), 1, dtype=torch.float32)
            denom = sigma * np.sqrt(np.pi * 2)
            target_y[kp_idx] = (torch.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2))) / denom
            target_x[kp_idx] = (torch.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) / denom

        return target_y, target_x, target_weight

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

    def evaluate(self,
                 preds_per_image: dict[int, list[torch.Tensor]],
                 epoch: int,
                 yx2xy: bool = True
                 ) -> tuple[OrderedDict[str,float], float]:
        """Calculate average precision using ground truth bbox."""
        pred_dump = []
        eval_target_image_ids = set()

        for image_id, preds in preds_per_image.items():
            eval_target_image_ids.add(image_id)
            for pred in preds:
                # (17,2) pred -> (17,3)
                kp = torch.zeros(17, 3, dtype=torch.int)
                # cocoapi uses xy coordinates.
                if yx2xy:
                    kp[:,0] = pred[:,1]
                    kp[:,1] = pred[:,0]
                else:
                    kp[:,0:2] = pred
                # These are dummy visibility score. cocoapi evaluation ignores pred vis score.
                kp[:,2] = (kp.sum(dim=1)>0).int()
                v = {"image_id": image_id,
                     "category_id": 1,     # category==1 -> person
                     "score": 1.0,         # dummy value
                     "keypoints": kp.flatten().tolist()}
                pred_dump.append(v)

        dump_dir = ".pred_dumps"
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        pred_dump_fname = f"{dump_dir}/pred_dump_epoch{epoch}"
        with open(pred_dump_fname, 'wb') as fout:
            fout.write(orjson.dumps(pred_dump))
        logger.info(f"Dumps validation data prediction results: {pred_dump_fname}")
        with redirect_stdout(None):  # mute the cocoapi internal print methods.
            coco_gt = COCO(self.anno_fname)
            coco_dt = coco_gt.loadRes(pred_dump_fname)
            coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
            coco_eval.params.imgIds = list(eval_target_image_ids)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)',
                       'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        eval_stats = OrderedDict()
        for i, name in enumerate(stats_names):
            eval_stats[name] = float(coco_eval.stats[i])

        # if os.path.exists(pred_dump_fname):
        #    os.remove(pred_dump_fname)

        return eval_stats, eval_stats['AP']
