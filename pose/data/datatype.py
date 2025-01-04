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

    def __repr__(self):
        return self._to_string()

    def __str__(self):
        return self._to_string()

    def _to_string(self) -> str:
        return f"y({self.y}) x({self.x}) h({self.h}) w({self.w})"

    def get_yxhw(self) -> tuple[int, int, int, int]:
        return self.y, self.x, self.h, self.w


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
            coco_kp = torch.tensor(anno, dtype=torch.int32).reshape(self.num_kp, 3)

            self._kp = torch.zeros(self.num_kp, 2, dtype=torch.int32)  # 2 for (y,x)
            # Convert COCO dataset coordinate system (x,y) to row-major (y,x)
            self._kp[:,0] = coco_kp[:,1]
            self._kp[:,1] = coco_kp[:,0]

            self._vis = coco_kp[:,2]
            self._vis_level = 2  # default visibility level
        else:
            raise Exception(f"Invalid data format: {data_format}")

    def is_ith_kp_visible(self, idx: int) -> bool:
        return bool(self._vis[idx] >= self._vis_level)

    def get_kp(self, coord: str = "yx") -> torch.Tensor:
        if coord == "yx":
            return self._kp
        elif coord == "xy":
            ret = torch.zeros(self.num_kp, 2)
            ret[:,0] = self._kp[:,1]
            ret[:,1] = self._kp[:,0]
            return ret
        else:
            raise ValueError(f"Invalid keypoint coordinate: {coord}")

    def get_vis(self) -> torch.Tensor:
        vis = copy.deepcopy(self._vis)
        vis[self._vis < self._vis_level] = 0
        return vis

    def get_bbox_size_and_center(self) -> tuple[int, int, int, int]:
        valid_keypoints = self._kp[self._vis >= self._vis_level]
        bh = max(valid_keypoints[:,0]) - min(valid_keypoints[:,0])
        bw = max(valid_keypoints[:,1]) - min(valid_keypoints[:,1])
        center_y = min(valid_keypoints[:,0]) + math.ceil(bh * 0.5)
        center_x = min(valid_keypoints[:,1]) + math.ceil(bw * 0.5)
        return bh, bw, center_y, center_x

    def update_kp(self,
                  idx: int,
                  coordinate: torch.Tensor,  # (y,x)
                  visibility: int
                  ) -> None:
        self._kp[idx] = coordinate
        self._vis[idx] = visibility

    def delete_kp(self, idx: int) -> None:
        self._kp[idx] = torch.tensor([0, 0])
        self._vis[idx] = 0

    def num_visible(self) -> int:
        return int(torch.sum(self._vis >= self._vis_level))

    def num_upperbody_visible(self) -> int:
        return int(torch.sum(self._vis[:self.upperbody_idx] >= self._vis_level))

    def num_lowerbody_visible(self) -> int:
        return int(torch.sum(self._vis[self.lowerbody_idx:] >= self._vis_level))

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
            y, x = self._kp[i]
            new_coord = [y, image_width - x]
            ret.update_kp(i, torch.Tensor(new_coord), int(self._vis[i]))
        return ret

    def rotate(self,
               angle: int,            # angle in degrees
               center: list[int],     # (y,x)
               image_size: list[int]  # (h,w)
               ) -> KeyPoints:
        theta = math.pi * angle / 180  # degrees to radians
        M = torch.tensor([[math.cos(theta), math.sin(theta)],
                          [-1 * math.sin(theta), math.cos(theta)]])
        c = torch.tensor(center)
        ret = copy.deepcopy(self)
        for i in range(self.num_kp):
            if ret.is_ith_kp_visible(i) is False:
                continue
            dst_vec = torch.matmul(M, (self._kp[i] - c).float()).int() + c
            if 0 <= dst_vec[0] < image_size[0] and 0 <= dst_vec[1] < image_size[1]:
                ret.update_kp(i, dst_vec, int(self._vis[i]))
            else:
                # keypoints outside of the image size
                ret.delete_kp(i)
        return ret

    def stretch(self,
                bbox: BBox,
                target_size: list[int],  # (h, w)
                ) -> KeyPoints:
        by, bx, bh, bw = bbox.get_yxhw()
        origin = torch.tensor([by, bx])
        th, tw = target_size
        M = torch.tensor([[float(th)/bh, 0],  # scaling linear transform
                          [0, float(tw)/bw]], dtype=torch.float32)
        ret = copy.deepcopy(self)
        for i in range(self.num_kp):
            if ret.is_ith_kp_visible(i) is False:
                continue
            dst_vec = torch.matmul(M, (self._kp[i] - origin).float()).int()
            if 0 <= dst_vec[0] < th and 0 <= dst_vec[1] < tw:
                ret.update_kp(i, dst_vec, int(self._vis[i]))
            else:
                # keypoints outside of the image size
                ret.delete_kp(i)
        return ret
