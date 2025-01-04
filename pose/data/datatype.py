from __future__ import annotations

import copy
import math

import torch


class BBox(object):
    def __init__(self, y: int, x: int, h:int, w:int) -> None:
        """
        y: left upper y coordinate
        x: left upper x coordinate
        h: height of the bounding box
        w: widht of the bounding box
        """
        self.y = y
        self.x = x
        self.h = h
        self.w = w

    def get_yxhw(self) -> tuple[int, int, int, int]:
        return self.y, self.x, self.h, self.w


class KeyPoints(object):
    num_kp = 17         # COCO dataset
    upperbody_idx = 13  # 13: L-knee
    lowerbody_idx = 11  # 11: L-hip

    def __init__(self, anno: list[float]) -> None:
        # COCO dataset keypoints annotation format: a list of length 51(=17x3)
        # 17x3: 17 for 17 keypoints, 3 for (x, y, v(isibility))
        # v=0: not labeled, v=1: labeled but not visible, and v=2: labeled and visible.
        kp = torch.tensor(anno, dtype=torch.int32).reshape(self.num_kp,3)
        self._kp = kp[:,:2]
        self._vis = kp[:,2]
        self._vis_level = 0  # default visibility level

    def is_ith_kp_visible(self, idx: int) -> bool:
        return bool(self._vis[idx] > self._vis_level)

    def get_kp(self) -> torch.Tensor:
        return self._kp

    def get_vis(self) -> torch.Tensor:
        return self._vis

    def get_bbox_size_and_center(self) -> tuple[int, int, int, int]:
        valid_keypoints = self._kp[self._vis > self._vis_level]
        bw = max(valid_keypoints[:,0]) - min(valid_keypoints[:,0])
        bh = max(valid_keypoints[:,1]) - min(valid_keypoints[:,1])
        center_x = min(valid_keypoints[:,0]) + math.ceil(bw * 0.5)
        center_y = min(valid_keypoints[:,1]) + math.ceil(bh * 0.5)
        return bw, bh, center_x, center_y

    def overwrite_kp(self,
                     idx: int,
                     coordinate: torch.Tensor,  # [x, y]
                     visibility: int
                     ) -> None:
        self._kp[idx] = coordinate
        self._vis[idx] = visibility

    def delete_kp(self, idx: int) -> None:
        self._kp[idx] = torch.tensor([0, 0])
        self._vis[idx] = 0

    def num_visible(self) -> int:
        return int(torch.sum(self._vis > self._vis_level))

    def num_upperbody_visible(self) -> int:
        return int(torch.sum(self._vis[:self.upperbody_idx] > self._vis_level))

    def num_lowerbody_visible(self) -> int:
        return int(torch.sum(self._vis[self.lowerbody_idx:] > self._vis_level))

    def remove_upperbody_kps(self) -> KeyPoints:
        ret = copy.deepcopy(self)
        for i in range(self.lowerbody_idx):
            if ret.is_ith_kp_visible(i) is False:
                continue
            ret.delete_kp(i)
        return ret

    def remove_lowerbody_kps(self) -> KeyPoints:
        ret = copy.deepcopy(self)
        for i in range(self.upperbody_idx, self.num_kp):
            if ret.is_ith_kp_visible(i) is False:
                continue
            ret.delete_kp(i)
        return ret

    def hflip(self, image_width: int) -> KeyPoints:
        ret = copy.deepcopy(self)
        for i in range(self.num_kp):
            if ret.is_ith_kp_visible(i) is False:
                continue
            x, y = self._kp[i]
            new_coord = [image_width - x, y]
            ret.overwrite_kp(i, torch.Tensor(new_coord), int(self._vis[i]))
        return ret

    def rotate(self,
               angle: int,            # angle in degrees
               center: list[int],     # (x, y)
               image_size: list[int]  # (w, h)
               ) -> KeyPoints:
        theta = math.pi * angle / 180  # to radian
        M = torch.tensor([[math.cos(theta), -1 * math.sin(theta)],
                          [math.sin(theta), math.cos(theta)]])
        c = torch.tensor(center)
        ret = copy.deepcopy(self)
        for i in range(self.num_kp):
            if ret.is_ith_kp_visible(i) is False:
                continue
            dst_vec = torch.matmul(M, (self._kp[i] - c).float()).int() + c
            if 0 <= dst_vec[0] < image_size[0] and 0 <= dst_vec[1] < image_size[1]:
                ret.overwrite_kp(i, dst_vec, int(self._vis[i]))
            else:
                # keypoints outside of the image size
                ret.delete_kp(i)
        return ret

    def stretch(self,
                bbox: BBox,
                target_size: list[int],
                ) -> KeyPoints:
        by, bx, bh, bw = bbox.get_yxhw()
        origin = torch.tensor([bx, by])
        th, tw = target_size
        M = torch.tensor([[float(tw)/bw, 0],  # scaling linear transform
                          [0, float(th)/bh]], dtype=torch.float32)
        ret = copy.deepcopy(self)
        for i in range(self.num_kp):
            if ret.is_ith_kp_visible(i) is False:
                continue
            dst_vec = torch.matmul(M, (self._kp[i] - origin).float()).int()
            if 0 <= dst_vec[0] < tw and 0 <= dst_vec[1] < th:
                ret.overwrite_kp(i, dst_vec, int(self._vis[i]))
            else:
                # keypoints outside of the image size
                ret.delete_kp(i)
        return ret
