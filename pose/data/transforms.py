#!/usr/bin/env python3
import cv2
import numpy as np
import numpy.typing as npt
import torch

from pose.data.datatype import BBox


def get_affine_transform(bbox: BBox,
                         scale: float,
                         rotate: float,
                         out_size_wh: list[float],  # (w,h)
                         ) -> npt.NDArray:
    """
    Since cv2.getAffineTransform() uses (x,y) coordinates,
    so this function uses (x,y) coordinates.
    """
    dst_w, dst_h = out_size_wh
    aspect_ratio = dst_w * 1.0 / dst_h

    w, h = map(float, bbox.size_wh())
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale_wh = np.array([w * scale, h * scale], dtype=np.float32)

    center_xy = np.array(bbox.center_xy(), dtype=np.float32)

    src_w = scale_wh[0]

    rotate_rad = np.pi * rotate / 180
    src_dir = get_rotated_point_xy([0, src_w * -0.5], rotate_rad)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    dst_center_xy = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    src[0, :] = center_xy
    src[1, :] = center_xy + src_dir
    dst[0, :] = dst_center_xy
    dst[1, :] = dst_center_xy + dst_dir

    src[2:, :] = get_affine_triangle_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_affine_triangle_3rd_point(dst[0, :], dst[1, :])

    return cv2.getAffineTransform(src, dst)


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
                                      real_bbox: list[float],     # (x,y,w,h)
                                      pred_bbox_size: list[float] # (w,h)
                                      ) -> torch.Tensor:
    ret = torch.zeros_like(kp)
    rx, ry, rw, rh = real_bbox
    pw, ph = pred_bbox_size
    ret[:,0] = kp[:,0] * pw**-1 * rw
    ret[:,1] = kp[:,1] * ph**-1 * rh
    ret[ret.sum(dim=1)>0,0] += rx
    ret[ret.sum(dim=1)>0,1] += ry
    return ret
