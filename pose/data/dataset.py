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
        self.image_meta = {}
        for _img in anno['images']:
            v = {}
            for att in ['file_name', 'height', 'width']:
                v[att] = _img[att]
            self.image_meta[_img['id']] = v
        logger.info(f"Num images: {len(self.image_meta):,d}")

        # load keypoint data
        self.keypoints = []
        for _anno in anno['annotations']:
            if _anno['iscrowd'] != 0 or _anno['category_id'] != 1:  # category_id==1: person
                continue
            v = {'keypoints': _anno['keypoints'],
                 'bbox': _anno['bbox'],
                 'image_id': _anno['image_id'],
                 'id': _anno['id']}
            self.keypoints.append(v)
        logger.info(f"Num keypoint data samples: {len(self.keypoints):,d}")

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return self.keypoints[idx]

    def get_image_meta(self, image_idx):
        return self.image_meta[image_idx]
