#!/usr/bin/env python3
import math

import torch
from loguru import logger
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2.functional import affine, crop, resize

from pose.data.datatype import BBox


class RescaleImgDiv255(object):
    def __call__(self, pic: torch.Tensor) -> torch.Tensor:
        if pic.dtype != torch.uint8:
            raise ValueError("dtype should be torch.uint8")
        return pic.div(255)


def rotate_image(image: torch.Tensor,
                 rotate: float,
                 center: list[float]  # (y,x)
                 ) -> torch.Tensor:
    # Change center coordinate (y,x) -> (x,y)
    # torchvision.transforms.affine() uses xy coordinate.
    center_xy = [center[1], center[0]]
    return affine(image, rotate, translate=[0,0], scale=1.0, shear=[0,0],
                  interpolation=InterpolationMode.BILINEAR, center=center_xy)


def crop_image(image: torch.Tensor,
               scale: float,
               center: list[float],        # (y, x)
               bbox_size: list[int],       # (h, w)
               out_image_size: list[int],  # (h, w)
               ) -> tuple[torch.Tensor, BBox]:
    cy, cx = center
    _bh, _bw = bbox_size
    oh, ow = out_image_size
    aspect_ratio = float(ow) / oh

    bh = math.floor(_bh * scale)
    bw = math.floor(_bw * scale)

    if bw > bh * aspect_ratio:
        bh = math.floor(bw * (aspect_ratio ** -1))
    elif bw < bh * aspect_ratio:
        bw = math.floor(bh * aspect_ratio)

    top = math.floor(cy - 0.5*bh)
    left = math.floor(cx - 0.5*bw)

    image = crop(image, top=top, left=left, height=bh, width=bw)
    image = resize(image, size=out_image_size,
                   interpolation=InterpolationMode.BILINEAR)

    new_bbox = BBox(y=top, x=left, h=bh, w=bw)
    logger.debug(f"new bbox: {new_bbox}")

    return image, new_bbox


def transform_bbox_pred_to_image_pred(kp: torch.Tensor,           # (17,2(=y,x))
                                      real_bbox: list[float],     # (y,x,h,w)
                                      pred_bbox_size: list[float] # (h,w)
                                      ) -> torch.Tensor:
    ret = torch.zeros_like(kp)
    ry, rx, rh, rw = real_bbox
    ph, pw = pred_bbox_size
    ret[:,0] = kp[:,0] * ph**-1 * rh
    ret[:,1] = kp[:,1] * pw**-1 * rw
    ret[ret.sum(dim=1)>0,0] += ry
    ret[ret.sum(dim=1)>0,1] += rx
    return ret
