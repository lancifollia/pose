#!/usr/bin/env python3
import math
import os
import sys

import torch
from loguru import logger
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2.functional import affine, resize

sys.path.insert(0, os.path.dirname(__file__))
from pose.data.datatype import BBox


class RescaleImgDiv255(object):
    def __call__(self, pic: torch.Tensor) -> torch.Tensor:
        if pic.dtype != torch.uint8:
            raise ValueError("dtype should be torch.uint8")
        return pic.div(255)


def rotate_image(image: torch.Tensor,
                 rotate: float,
                 center: list[int]  # (y,x)
                 ) -> torch.Tensor:
    # Change center coordinate (y,x) -> (x,y)
    # torchvision.transforms.affine() uses xy coordinate.
    center_xy = [center[1], center[0]]
    return affine(image, rotate, translate=[0,0], scale=1.0, shear=[0,0],
                  interpolation=InterpolationMode.BILINEAR, center=center_xy)


def crop_and_resize_image(image: torch.Tensor,
                          scale: float,
                          center: list[int],         # (y, x)
                          src_box_size: list[int],   # (h, w)
                          out_image_size: list[int]  # (h, w)
                          ) -> tuple[torch.Tensor, BBox]:
    cy, cx = center
    sbh, sbw = src_box_size
    ih, iw = image.shape[1:]  # image.shape: (C,H,W)
    oh, ow = out_image_size
    aspect_ratio = float(ow) / oh

    logger.debug(f"original image size(c,h,w): {image.shape}")
    logger.debug(f"original box size(h,w): {sbh}, {sbw}")
    logger.debug(f"bbox center: {center}")

    # Resizing bbox & cropping
    bh = int(min(sbh * scale, 2 * (ih - cy), 2 * cy))
    logger.debug(f"bh=min({sbh * scale}, {2 * (ih - cy)}, {2 * cy})")
    bw = int(min(sbw * scale, 2 * (iw - cx), 2 * cx))
    logger.debug(f"bw=min({sbw * scale}, {2 * (iw - cx)}, {2 * cx})")
    if bw >= bh * aspect_ratio:
        bh = int(min(float(bw) * aspect_ratio**-1, 2 * (ih - cy), 2 * cy))
        bw = int(bh * aspect_ratio)
    else:
        bw = int(min(bh * aspect_ratio, 2 * (iw - cx), 2 * cx))
        bh = int(float(bw) * aspect_ratio**-1)
    # On vertically long images, focus on the top.
    y1 = max(min(cy - math.ceil(bh * 0.5),
                 cy - math.ceil(sbh * 0.5)), 0)
    x1 = max(cx - math.ceil(bw * 0.5), 0)
    new_bbox = BBox(y=y1, x=x1, h=bh, w=bw)
    logger.debug(f"new bbox: {new_bbox}")

    image = image[:, y1:y1+bh, x1:x1+bw]
    logger.debug(f"expected output size: {out_image_size}")
    logger.debug(f"cropped image size: {image.shape}")
    logger.debug(f"cropped image aspect ratio: {float(image.shape[2])/image.shape[1]}")
    cropped_aspect_ratio = float(image.shape[2])/image.shape[1]
    if math.fabs(aspect_ratio - cropped_aspect_ratio) / aspect_ratio > 0.05:
        raise RuntimeError(f"Invalid cropped aspect ratio: {cropped_aspect_ratio}")

    # Resizing cropped image
    image = resize(image, size=out_image_size,
                   interpolation=InterpolationMode.BILINEAR)

    return image, new_bbox
