#!/usr/bin/env python3

from typing import List

import cv2
import numpy as np
import numpy.typing as npt
import torch


def flip_back_simcc(output_flipped: npt.NDArray,
                    flip_pairs: List[List[int]],
                    axis: str
                    ) -> npt.NDArray:
    assert output_flipped.ndim == 3,\
        "output_flipped should be [batch_size, num_keypoints, axis_resolution * simcc_scale_factor]"
    assert axis in ["x", "y"]

    if axis == "x":
        output_flipped = output_flipped[:, :, ::-1]

    for pair in flip_pairs:
        tmp = output_flipped[:, pair[0], :].copy()
        output_flipped[:, pair[0], :] = output_flipped[:, pair[1], :]
        output_flipped[:, pair[1], :] = tmp

    return output_flipped



def get_affine_transform(bbox_center: npt.NDArray,
                         bbox_scale: npt.NDArray,
                         rotate: float,
                         out_size_wh: list[float],  # (w,h)
                         inverse=False,
                         ) -> npt.NDArray:
    """
    Since cv2.getAffineTransform() uses (x,y) coordinates,
    so this function uses (x,y) coordinates.
    """
    src_w = bbox_scale[0] * 200   # pixel std

    dst_w, dst_h = out_size_wh

    rotate_rad = np.pi * rotate / 180
    src_dir = get_rotated_point_xy([0, src_w * -0.5], rotate_rad)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    dst_center_xy = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    src[0, :] = bbox_center
    src[1, :] = bbox_center + src_dir
    dst[0, :] = dst_center_xy
    dst[1, :] = dst_center_xy + dst_dir

    src[2:, :] = get_affine_triangle_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_affine_triangle_3rd_point(dst[0, :], dst[1, :])

    if inverse:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)

    return trans


def get_rotated_point_xy(point_xy: list[float],  # (x,y)
                         rotate_rad: float
                         ) -> npt.NDArray:
    sin, cos = np.sin(rotate_rad), np.cos(rotate_rad)

    result = np.array([0.0, 0.0], dtype=np.float32)
    result[0] = point_xy[0] * cos - point_xy[1] * sin
    result[1] = point_xy[0] * sin + point_xy[1] * cos

    return result


def get_affine_triangle_3rd_point(catheti: npt.NDArray,
                                  hypotenuse: npt.NDArray,
                                  ) -> npt.NDArray:
    """Get another catheti of the right triangle"""
    direction = catheti - hypotenuse
    return hypotenuse + np.array([-direction[1], direction[0]], dtype=np.float32)


def transform_bbox_pred_to_image_pred(kp: torch.Tensor,           # (17,2(=x,y))
                                      center: npt.NDArray,
                                      scale: npt.NDArray,
                                      pred_bbox_size: list[float] # (w,h)
                                      ) -> torch.Tensor:
    ret = np.zeros(kp.shape)
    trans = get_affine_transform(bbox_center=center,
                                 bbox_scale=scale,
                                 rotate=0.0,
                                 out_size_wh=pred_bbox_size,
                                 inverse=True)
    for p in range(kp.shape[0]):
        pt = kp[p]
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(trans, new_pt)
        ret[p] = new_pt
    return torch.from_numpy(ret)


def transform_bbox_pred_to_image_pred_backup(kp: torch.Tensor,           # (17,2(=x,y))
                                             real_bbox: list[float],     # (x,y,w,h)
                                             pred_bbox_size: list[float] # (w,h)
                                             ) -> torch.Tensor:
    ret = torch.zeros_like(kp)
    rx, ry, rw, rh = real_bbox
    pw, ph = pred_bbox_size
    ret[:,0] = kp[:,0] * (float(pw)**-1) * rw
    ret[:,1] = kp[:,1] * (float(ph)**-1) * rh
    ret[ret.sum(dim=1)>0,0] += rx
    ret[ret.sum(dim=1)>0,1] += ry
    return ret
