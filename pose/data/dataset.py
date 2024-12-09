#!/usr/bin/env python3

"""
Dataset modules
"""

import orjson as json
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, anno_fname, image_dir):
        self.image_dir = image_dir
        with open(anno_fname) as _anno_fin:
            anno = json.loads(_anno_fin)

        # load images
        self.images = {}
        for _img in anno['images']:
            v = {}
            for att in ['file_name', 'height', 'width']:
                v[att] = _img[att]
            self.images[_img['id']] = v

        # load keypoints
        self.keypoints = {}
        for _anno in anno['annotations']:
            if _anno['iscrowd'] != 0:
                continue
            #TODO

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        pass
