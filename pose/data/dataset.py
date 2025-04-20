"""
Dataset modules
"""

import copy
import os
import pickle
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional, NamedTuple

import cv2
import numpy as np
import orjson
import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torch.utils.data import Dataset
from yacs.config import CfgNode

from pose.data.datatype import BBox, KeyPoints
from pose.data.transforms import get_affine_transform


class ImageMeta(NamedTuple):
    fpath: str  # image file path
    w: int      # width of the image
    h: int      # height of the image


class DataSample(NamedTuple):
    keypoints: KeyPoints
    bbox: BBox
    image_id: int
    id: int


class CocoDataset(Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 is_train: bool,
                 is_debug: bool = False,
                 transform: Optional[Any] = None
                 ) -> None:
        self._init_coco_info()
        self.is_train = is_train
        data_cfg = cfg.DATA_TRAIN if is_train else cfg.DATA_VAL
        self.is_debug = is_debug

        self.anno_fname = data_cfg.ANNOTATION_FNAME
        with open(self.anno_fname) as _anno_fin:
            anno = orjson.loads(_anno_fin.read())
        logger.info(f"Annotation fname: {self.anno_fname}")
        self.image_dir = data_cfg.IMAGE_DIR
        logger.info(f"Image file directory: {self.image_dir}")

        self.nn_input_size_wh = cfg.MODEL.INPUT_SIZE_WH
        logger.info(f"NN input size(width, height): {self.nn_input_size_wh}")
        self.aspect_ratio = self.nn_input_size_wh[0] * 1.0 / self.nn_input_size_wh[1]
        logger.info(f"Aspect ratio(=width/height): {self.aspect_ratio:.4f}")
        self.bbox_scale_factor = 1.25
        logger.info(f"BBox scale factor: {self.bbox_scale_factor:.4f}")

        self.min_keypoints = cfg.DATA_COMMON.MIN_KEYPOINTS

        self.simcc_split_factor = cfg.MODEL.EXTRA.SIMCC_SPLIT_FACTOR
        self.simcc_sigma = cfg.MODEL.EXTRA.SIMCC_SIGMA

        self.transform = transform

        data_aug_cfg = data_cfg.DATA_AUG
        if data_aug_cfg is None:
            self.do_data_augment = False
            self.flip_proba = 0.0
            self.rotate_proba = 0.0
            self.rotate_factor = 0
            self.scale_factor = 0.0
        else:
            self.do_data_augment = True
            self.flip_proba = data_aug_cfg.FLIP_PROBA
            self.rotate_proba = data_aug_cfg.ROTATE_PROBA
            self.rotate_factor = data_aug_cfg.ROTATE_FACTOR
            self.scale_factor = data_aug_cfg.SCALE_FACTOR
        logger.info(f"Data augmentation: {self.do_data_augment}")

        # load image meta data
        self._image_meta = {}
        for _image in anno['images']:
            m = ImageMeta(fpath=os.path.join(self.image_dir, _image['file_name']),
                          w=_image['width'], h=_image['height'])
            self._image_meta[_image['id']] = m
        logger.info(f"Num images: {len(self._image_meta):,d}")

        # load keypoint data
        logger.info(f"Num raw annotations: {len(anno['annotations']):,d}")
        num_skip_crowd = 0
        num_skip_not_human = 0
        num_skip_insufficient_keypoints = 0
        num_skip_invalid_bbox = 0
        self._dataset = list[DataSample]()

        # TODO: need to delete after verification
        # anno_set = set([183020, 183030, 183050, 183062, 183092, 183103, 183141, 183194, 183263, 183277])
        # anno_set = set([183020])

        for _anno in anno['annotations']:
            # TODO: need to delete after verification
            # if _anno['id'] not in anno_set:
            #    continue

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
            iw = image_meta.w
            ih = image_meta.h
            x, y, bw, bh = _anno['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(iw - 1, x1 + max(0, bw - 1))
            y2 = min(ih - 1, y1 + max(0, bh - 1))
            if _anno['area'] > 0 and x2 >= x1 and y2 >= y1:
                bbox = BBox(x=x1, y=y1, w=x2-x1, h=y2-y1)
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

    def get_item(self,
                 idx: int,
                 hflip: bool = False,
                 rotate: int = 0,
                 scale: float = 1.0
                 ) -> tuple[Any, Tensor, Tensor, Tensor, Dict]:
        sample = self.get_sample(idx)
        image_meta = self.get_image_meta(sample.image_id)
        image_path = image_meta.fpath

        bbox = sample.bbox.transform(self.aspect_ratio, self.bbox_scale_factor * scale)

        # shape: (height, width, color channels)
        image_numpy = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if image_numpy is None:
            logger.error(f"=> fail to read {image_path}")
            raise ValueError(f"Fail to read {image_path}")

        keypoints = sample.keypoints

        if hflip is True:
            image_numpy = image_numpy[:, ::-1, :]
            bbox = bbox.hflip(image_meta.w)
            keypoints = keypoints.hflip(image_meta.w)

        trans = get_affine_transform(bbox, rotate, self.nn_input_size_wh)
        input_phase1 = cv2.warpAffine(
            image_numpy,
            trans,
            (int(self.nn_input_size_wh[0]), int(self.nn_input_size_wh[1])),
            flags=cv2.INTER_LINEAR)

        keypoints = keypoints.affine_transform(trans)

        if self.transform is not None:
            input_phase2 = self.transform(input_phase1)
        else:
            input_phase2 = input_phase1

        target_x, target_y, target_weight = self._generate_simcc_label(keypoints)
        meta = {"image_id": torch.tensor(sample.image_id).type(torch.long),
                "anno_id": torch.tensor(sample.id).type(torch.long),
                "origin_bbox": torch.tensor(sample.bbox.get_xywh()).type(torch.float),
                "bbox": torch.tensor(bbox.get_xywh()).type(torch.float),
                "keypoints": keypoints.get_kps(),
                "vis": keypoints.get_vis()}

        if self.is_debug:
            probe = {"bbox": bbox.get_xywh(), "image_numpy": image_numpy,
                     "origin_kps": keypoints.get_kps(), "origin_vis": keypoints.get_vis(),
                     "scale": scale, "rotate": rotate, "center": bbox.get_center_xy(),
                     "affine_trans": trans, "input_phase1": input_phase1, "input_phase2": input_phase2,
                     "final_kps": keypoints.get_kps(), "final_vis": keypoints.get_vis(),
                     "anno_id": sample.id, "image_id": sample.image_id,
                     "target_x": target_x, "target_y": target_y, "target_weight": target_weight}
            category = "train" if self.is_train else "val"
            with open(f"debug/pose_nn_input_{category}_annoid{sample.id}.pkl", "wb") as fout:
                pickle.dump(probe, fout, pickle.HIGHEST_PROTOCOL)

        return input_phase2, target_x, target_y, target_weight, meta

    def __getitem__(self, idx: int)\
        -> tuple[Any, Tensor, Tensor, Tensor, Dict]:
        hflip = False
        rotate = 0
        scale = 1.0

        if self.do_data_augment:
            if np.random.random() <= self.flip_proba:
                hflip = True
            if np.random.random() <= self.rotate_proba:
                rotate = np.clip(np.random.randn() * self.rotate_factor,
                                 self.rotate_factor * -2, self.rotate_factor * 2)
            scale = np.clip(np.random.randn() * self.scale_factor + 1,
                            1 - self.scale_factor, 1 + self.scale_factor)

        return self.get_item(idx, hflip, rotate, scale)

    def _generate_simcc_label(self, keypoints: KeyPoints)\
        -> tuple[Tensor, Tensor, Tensor]:
        target_weight = torch.zeros((self.num_keypoints, 1), dtype=torch.float32)
        target_weight[keypoints.get_vis() > 0] = 1.0

        w, h = self.nn_input_size_wh
        sf = self.simcc_split_factor

        target_x = np.zeros((self.num_keypoints, int(w * sf)), dtype=np.float32)
        target_y = np.zeros((self.num_keypoints, int(h * sf)), dtype=np.float32)

        sigma = self.simcc_sigma
        buffer = sigma * 3
        for kp_idx in range(self.num_keypoints):
            kx, ky = keypoints.get_kp(kp_idx)

            # Check that any part of the gaussian is in-bounds
            ulx = kx - buffer
            uly = ky - buffer
            brx = kx + buffer + 1
            bry = ky + buffer + 1
            if ulx < 0 or uly < 0 or brx > w or bry > h:
                target_weight[kp_idx] = 0.0

            if target_weight[kp_idx] < 0.5:
                continue

            # Space-aware Gaussian label smoothing
            mu_x = kx * sf
            mu_y = ky * sf
            x = np.arange(0, int(w * sf), 1, dtype=np.float32)
            y = np.arange(0, int(h * sf), 1, dtype=np.float32)
            denom = sigma * np.sqrt(np.pi * 2)
            target_x[kp_idx] = (np.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) / denom
            target_y[kp_idx] = (np.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2))) / denom

        return torch.from_numpy(target_x), torch.from_numpy(target_y), target_weight

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
                 epoch: int
                 ) -> OrderedDict[str,float]:
        """Calculate average precision using ground truth bbox."""
        pred_dump = []
        eval_target_image_ids = set()

        for image_id, preds in preds_per_image.items():
            eval_target_image_ids.add(image_id)
            for pred in preds:
                # (17,2) pred -> (17,3)
                kp = torch.zeros(17, 3, dtype=torch.float32)
                # cocoapi uses xy coordinates.
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

        return eval_stats
