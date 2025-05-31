"""
Dataset modules
"""

import copy
import os
import pickle
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import orjson as json
import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torch.utils.data import Dataset
from yacs.config import CfgNode

from pose.data.datatype import DataSample, KeyPoints
from pose.data.transforms import get_affine_transform


class CocoDataset(Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 is_train: bool,
                 is_debug: bool = False,
                 transform: Optional[Any] = None
                 ) -> None:
        self.is_train = is_train
        data_cfg = cfg.DATA_TRAIN if is_train else cfg.DATA_VAL
        self.is_debug = is_debug

        self.nn_input_size_wh = cfg.MODEL.INPUT_SIZE_WH
        logger.info(f"NN input size(width, height): {self.nn_input_size_wh}")
        self.aspect_ratio = self.nn_input_size_wh[0] * 1.0 / self.nn_input_size_wh[1]
        logger.info(f"Aspect ratio(=width/height): {self.aspect_ratio:.4f}")
        self.bbox_scale_factor = 1.25
        logger.info(f"BBox scale factor: {self.bbox_scale_factor:.4f}")

        self._init_coco_data(data_cfg)

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

    def _init_coco_data(self, data_cfg: CfgNode) -> None:
        self.data_format = "COCO"

        self.anno_fname = data_cfg.ANNOTATION_FNAME
        logger.info(f"Annotation fname: {self.anno_fname}")
        self.coco = COCO(self.anno_fname)

        self.image_dir = data_cfg.IMAGE_DIR
        logger.info(f"Image file directory: {self.image_dir}")

        # Deal with class names
        categories = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + categories
        logger.info(f"COCO data classes: {self.classes}")
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.class_to_coco_ind = dict(zip(categories, self.coco.getCatIds()))
        self.coco_ind_to_class_ind = dict(
            [
                (self.class_to_coco_ind[cls], self.class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # Load image file names (image_id: int)
        if data_cfg.USE_TARGET_IMAGES and len(data_cfg.TARGET_IMAGES) > 0:
            self.target_image_ids = data_cfg.TARGET_IMAGES
            logger.warning(f"Retrict target images(size: {len(self.target_image_ids):,}): {self.target_image_ids}")
        else:
            self.target_image_ids = self.coco.getImgIds()
        self.num_images = len(self.target_image_ids)
        logger.info(f"number of target images: {self.num_images:,} (is_train: {self.is_train})")

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

        self._dataset = self._load_coco_keypoint_annotations()
        logger.info(f"Load {len(self._dataset):,} annotations")

    def __len__(self) -> int:
        return len(self._dataset)

    def _load_coco_keypoint_annotations(self) -> List[DataSample]:
        """Load ground truth bbox and keypoints"""
        gt_dataset = []
        for target_image_id in self.target_image_ids:
            gt_dataset.extend(self._load_coco_keypoint_annotation_kernel(target_image_id))
        return gt_dataset

    def _load_coco_keypoint_annotation_kernel(self, image_id: int) -> List[DataSample]:
        """
        coco annotation:
            ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param image_id: coco image id
        :return: dataset
        """
        image_meta = self.coco.loadImgs(image_id)[0]
        image_width = image_meta['width']
        image_height = image_meta['height']

        anno_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        objs = self.coco.loadAnns(anno_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((image_width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((image_height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)

        ret = []
        for obj in valid_objs:
            cls = self.coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:  # cls:1 => person
                continue

            # anno_id filter
            # if obj['id'] != 1202318:
            #    continue
            # if self.is_train and image_id not in [184613, 522418, 554625]:
            #     continue

            # ignore objects without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            kp = KeyPoints(anno=obj['keypoints'], data_format=self.data_format)
            bbox_center, bbox_scale = self.bbox_to_center_scale(*obj['clean_bbox'])
            ret.append(DataSample(image_id=image_id, anno_id=obj['id'],
                                  image_width=image_width, image_height=image_height,
                                  image_fpath=os.path.join(self.image_dir, image_meta['file_name']),
                                  bbox_center=bbox_center, bbox_scale=bbox_scale, keypoints=kp
                                  ))
        return ret

    def bbox_to_center_scale(self, x: int, y: int, w: int, h: int
                             ) -> Tuple[npt.NDArray, npt.NDArray]:
        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w * 0.5

        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0 / 200, h * 1.0 / 200], dtype=np.float32)
        scale = scale * self.bbox_scale_factor
        return center, scale

    def get_item(self,
                 idx: int,
                 hflip: bool = False,
                 rotate: int = 0,
                 scale: float = 1.0
                 ) -> tuple[Any, Tensor, Tensor, Tensor, Dict]:
        sample = self.get_sample(idx)
        image_path = sample.image_fpath

        # shape: (height, width, color channels)
        image_numpy = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if image_numpy is None:
            logger.error(f"=> fail to read {image_path}")
            raise ValueError(f"Fail to read {image_path}")

        bbox_center = sample.bbox_center
        bbox_scale = sample.bbox_scale
        keypoints = sample.keypoints

        if hflip is True:
            image_numpy = image_numpy[:, ::-1, :]
            keypoints = keypoints.hflip(sample.image_width)
            bbox_center[0] = image_numpy.shape[1] - bbox_center[0] - 1

        trans = get_affine_transform(bbox_center, bbox_scale, rotate, self.nn_input_size_wh)
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
                "anno_id": torch.tensor(sample.anno_id).type(torch.long),
                "bbox_center": torch.from_numpy(bbox_center),
                "bbox_scale": torch.from_numpy(bbox_scale),
                "keypoints": torch.from_numpy(keypoints.get_kps()),
                "vis": torch.from_numpy(keypoints.get_vis()),
                }

        if self.is_debug:
            meta["trans"] = torch.from_numpy(trans)
            meta["input_phase1"] = torch.from_numpy(input_phase1)

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

    def _generate_simcc_label(self, keypoints: KeyPoints) -> tuple[Tensor, Tensor, Tensor]:
        target_weight = torch.zeros((self.num_keypoints, 1), dtype=torch.float32)
        target_weight[keypoints.get_vis() > 0] = 1.0

        w, h = self.nn_input_size_wh
        sf = self.simcc_split_factor

        target_x = np.zeros((self.num_keypoints, int(w * sf)), dtype=np.float32)
        target_y = np.zeros((self.num_keypoints, int(h * sf)), dtype=np.float32)

        label_smoothing_size = self.simcc_sigma * 3
        for kp_idx in range(self.num_keypoints):
            kx, ky = keypoints.get_kp(kp_idx)

            # Check that any part of the gaussian is in-bounds
            ulx = int(kx - label_smoothing_size)
            uly = int(ky - label_smoothing_size)
            brx = int(kx + label_smoothing_size + 1)
            bry = int(ky + label_smoothing_size + 1)
            if ulx >= w or uly >= h or brx < 0 or bry < 0:
                target_weight[kp_idx] = 0.0

            if target_weight[kp_idx] < 0.5:
                continue

            # Space-aware Gaussian label smoothing
            mu_x = kx * sf
            mu_y = ky * sf
            x = np.arange(0, int(w * sf), 1, dtype=np.float32)
            y = np.arange(0, int(h * sf), 1, dtype=np.float32)
            target_x[kp_idx] = (np.exp(- ((x - mu_x) ** 2) / (2 * self.simcc_sigma ** 2))) / (self.simcc_sigma * np.sqrt(np.pi * 2))
            target_y[kp_idx] = (np.exp(- ((y - mu_y) ** 2) / (2 * self.simcc_sigma ** 2))) / (self.simcc_sigma * np.sqrt(np.pi * 2))

        return torch.from_numpy(target_x), torch.from_numpy(target_y), target_weight

    def get_connected_keypoints(self, keypoint_idx: int) -> list[int]:
        ret = set()
        for a, b in self.skeleton:
            if a == keypoint_idx:
                ret.add(b)
            elif b == keypoint_idx:
                ret.add(a)
        return list(ret)

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
            fout.write(json.dumps(pred_dump))
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
