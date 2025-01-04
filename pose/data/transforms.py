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
                 center: list[int]
                 ) -> torch.Tensor:
    return affine(image, rotate, translate=[0,0], scale=1.0, shear=[0,0],
                  interpolation=InterpolationMode.BILINEAR, center=center)


def crop_and_resize_image(image: torch.Tensor,
                          scale: float,
                          center: list[int],
                          box_width: int,
                          box_height: int,
                          out_image_size: list[int]
                          ) -> tuple[torch.Tensor, BBox]:
    ih, iw = image.shape[1:]  # input image shape
    oh, ow = out_image_size
    aspect_ratio = float(ow) / oh

    # Resizing bbox & cropping
    logger.debug(f"center: {center}")
    bw = int(min(box_width * scale, 2 * (iw - center[0]), 2 * center[0]))
    logger.debug(f"bw, {box_width * scale}, {2 * (iw - center[0])}, {2 * center[0]}")
    bh = int(min(box_height * scale, 2 * (ih - center[1]), 2 * center[1]))
    logger.debug(f"bh, {box_height * scale}, {2 * (ih - center[1])}, {2 * center[1]}")
    if bw >= bh * aspect_ratio:
        bh = int(min(float(bw) * aspect_ratio**-1, 2 * (ih - center[1]), 2 * center[1]))
        bw = int(bh * aspect_ratio)
    else:
        bw = int(min(bh * aspect_ratio, 2 * (iw - center[0]), 2 * center[0]))
        bh = int(float(bw) * aspect_ratio**-1)
    x1 = max(center[0] - math.ceil(bw * 0.5), 0)
    # On vertically long images, focus on the top.
    y1 = max(min(center[1] - math.ceil(bh * 0.5),
                 center[1] - math.ceil(box_height * 0.5)), 0)
    logger.debug(f"original image size: {image.shape}")
    logger.debug(f"original box size: {box_width}, {box_height}")
    new_bbox = BBox(y1, x1, bh, bw)
    logger.debug(f"new bbox: {new_bbox}")

    image = image[:, y1:y1+bh, x1:x1+bw]
    logger.debug(f"expected output size: {out_image_size}")
    logger.debug(f"cropped image size: {image.shape}")
    logger.debug(f"cropped image aspect ratio: {float(image.shape[2])/image.shape[1]}")
    cropped_aspect_ratio = float(image.shape[2])/image.shape[1]
    if math.fabs(aspect_ratio - cropped_aspect_ratio) / aspect_ratio > 0.03:
        raise RuntimeError(f"Invalid cropped aspect ratio: {cropped_aspect_ratio}")

    # Resizing cropped image
    image = resize(image, size=out_image_size,
                   interpolation=InterpolationMode.BILINEAR)

    return image, new_bbox
