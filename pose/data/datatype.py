from __future__ import annotations

import copy
from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class DataSample(NamedTuple):
    image_id: int
    anno_id: int
    image_width: int
    image_height: int
    image_fpath: str
    bbox_center: npt.NDArray
    bbox_scale: npt.NDArray
    keypoints: KeyPoints


class KeyPoints(object):

    def __init__(self, anno: list[float], data_format: str) -> None:
        if data_format == "COCO":
            # General COCO dataset info
            self.num_kp = 17         # COCO dataset
            self.upperbody_idx = 13  # 13: L-knee
            self.lowerbody_idx = 11  # 11: L-hip

            # COCO dataset keypoints annotation format: a list of length 51(=17x3)
            # - 17x3: 17 for 17 keypoints, 3 for (x, y, v(isibility))
            # - v=0: not labeled, v=1: labeled but not visible, and v=2: labeled and visible.
            coco_kp = np.array(anno, dtype=np.float32).reshape(self.num_kp, 3)
            self._kp = coco_kp[:,:2]
            self._vis = coco_kp[:,2]
            self._vis_level = 1  # default visibility level
        else:
            msg = f"Only COCO dataset format supported yet. Invalid data format: {data_format}"
            raise Exception(msg)

    def is_ith_kp_visible(self, idx: int) -> bool:
        return bool(self._vis[idx] >= self._vis_level)

    def get_kp(self, idx: int) -> npt.NDArray:
        return self._kp[idx]

    def get_kps(self) -> npt.NDArray:
        return copy.deepcopy(self._kp)

    def get_vis(self) -> npt.NDArray:
        vis = copy.deepcopy(self._vis)
        vis[self._vis < self._vis_level] = 0.0
        return vis

    def update_kp(self,
                  idx: int,
                  pt_xy: npt.NDArray,  # (x,y)
                  visibility: float
                  ) -> None:
        self._kp[idx] = pt_xy
        self._vis[idx] = visibility

    def delete_kp(self, idx: int) -> None:
        self._kp[idx] = np.array([0.0, 0.0], dtype=np.float32)
        self._vis[idx] = 0.0

    def num_visible(self) -> int:
        return int(np.sum(self._vis >= self._vis_level))

    def hflip(self, image_width: int) -> KeyPoints:
        ret = copy.deepcopy(self)

        # Flip coordinates
        for i in range(self.num_kp):
            if ret.is_ith_kp_visible(i) is False:
                continue
            x, y = self._kp[i]
            new_coord = [image_width - x, y]
            ret.update_kp(i, np.array(new_coord, dtype=np.float32), float(self._vis[i]))

        # Swap left-right joints (Skip Nose(keypoint idx==0))
        for i in range(1, self.num_kp, 2):
            left_idx = i
            right_idx = i+1
            left_kp = copy.deepcopy(ret._kp[left_idx])
            left_vis = copy.deepcopy(ret._vis[left_idx])
            ret._kp[left_idx] = ret._kp[right_idx]
            ret._vis[left_idx] = ret._vis[right_idx]
            ret._kp[right_idx] = left_kp
            ret._vis[right_idx] = left_vis
        return ret

    def affine_transform(self, trans: npt.NDArray) -> KeyPoints:
        ret = copy.deepcopy(self)
        for i in range(self.num_kp):
            if ret.is_ith_kp_visible(i) is False:
                continue
            pt = ret._kp[i]
            new_pt = np.array([pt[0], pt[1], 1.]).T
            new_pt = np.dot(trans, new_pt)
            ret._kp[i] = new_pt
        return ret
