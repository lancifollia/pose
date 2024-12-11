#!/usr/bin/env python3

"""
Dataset modules
"""

import orjson as json
from loguru import logger
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, anno_fname, image_dir):
        self.image_dir = image_dir
        with open(anno_fname) as _anno_fin:
            anno = json.loads(_anno_fin.read())
        logger.info(f"Annotation fname: {anno_fname}")

        # load image meta data
        self.images = {}
        for _img in anno['images']:
            v = {}
            for att in ['file_name', 'height', 'width']:
                v[att] = _img[att]
            self.images[_img['id']] = v
        logger.info(f"Num images: {len(self.images):,d}")

        # load keypoint data
        self.keypoints = {}
        for _anno in anno['annotations']:
            if _anno['iscrowd'] != 0 or _anno['category_id'] != 1:  # category_id==1: person
                continue
            self.keypoints[_anno['id']] = {'keypoints': _anno['keypoints'],
                                           'image_id': _anno['image_id'],
                                           'bbox': _anno['bbox']}
        logger.info(f"Num keypoint data samples: {len(self.keypoints):,d}")

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        pass
