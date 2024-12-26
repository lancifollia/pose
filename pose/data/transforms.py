#!/usr/bin/env python3
import math

import torch
from loguru import logger
from torchvision.transforms.v2.functional import affine, resize
from torchvision.transforms.v2 import InterpolationMode



class RescaleImgDiv255(object):

    def __call__(self, pic):
        if type(pic) not in [torch.tensor, torch.Tensor]:
            raise ValueError("Only torch.tensor type is supported.")
        if pic.dtype != torch.uint8:
            raise ValueError("dtype should be torch.uint8")
        return pic.div(255)


def rotate_image(image, rotate, center):
    return affine(image, rotate, translate=[0,0], scale=1.0, shear=[0,0],
                  interpolation=InterpolationMode.BILINEAR, center=center)


def rotate_keypoints(keypoints, rotate, center, image_size):
    theta = math.pi * rotate / 180  # angle in degrees -> radian
    M = torch.tensor([[math.cos(theta), -1 * math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
    c = torch.tensor(center)
    dst_keypoints = torch.zeros(keypoints.shape, dtype=torch.int32)
    for i in range(keypoints.shape[0]):
        if keypoints[i,2] == 0:   # visibility
            continue
        dst_vec = torch.matmul(M, (keypoints[i,:2] - c).float()).int() + c
        if 0 <= dst_vec[0] < image_size[0] and 0 <= dst_vec[1] < image_size[1]:
            dst_keypoints[i,:2] = dst_vec
            dst_keypoints[i,2] = keypoints[i,2]
        else:
            # keypoints outside of the image size
            dst_keypoints[i] = torch.tensor([0, 0, 0])
    return dst_keypoints



def crop_image1(image, keypoints, scale, center,
               box_width, box_height, out_image_size):
    ih, iw = image.shape[1:]  # input image shape
    oh, ow = out_image_size
    aspect_ratio = float(ow) / oh

    if iw >= ih * aspect_ratio:
        max_iw = ih * aspect_ratio
        max_ih = ih
    else:
        max_ih = int(float(iw) * aspect_ratio**-1)
        max_iw = iw

    # Resizing bbox & cropping
    logger.info(f"center: {center}")
    bw = int(min(box_width * scale, max_iw))
    bh = int(min(box_height * scale, max_ih))
    if bw >= bh * aspect_ratio:
        bh = int(min(float(bw) * aspect_ratio**-1, max_ih))
        y1 = center[1] - math.ceil(bh * 0.5)
        if y1 + bh > ih:
            y1 = ih - bh - 1
        if y1 < 0:
            y1 = 0
        x1 = center[0] - math.ceil(bw * 0.5)
        if x1 + bw > iw:
            x1 = iw - bw - 1
        if x1 < 0:
            x1 = 0
    else:
        bw = int(min(bh * aspect_ratio, max_iw))
        x1 = center[0] - math.ceil(bw * 0.5)
        if x1 + bw > iw:
            x1 = iw - bw - 1
        if x1 < 0:
            x1 = 0
        y1 = center[1] - math.ceil(bh * 0.5)
        if y1 + bh > ih:
            y1 = ih - bh - 1
        if y1 < 0:
            y1 = 0

    logger.info(f"original image size: {image.shape}")
    logger.info(f"original box size: {box_width}, {box_height}")
    logger.info(f"x1({x1}), y1({y1}), bw({bw}), bh({bh})")
    image = image[:, y1:y1+bh, x1:x1+bw]
    logger.info(f"expected output size: {out_image_size}")
    logger.info(f"cropped image size: {image.shape}")
    logger.info(f"cropped image aspect ratio: {float(image.shape[2])/image.shape[1]}")

    # Resizing cropped image
    image = resize(image, size=out_image_size,
                   interpolation=InterpolationMode.BILINEAR)

    # Transform keypoints
    crop_origin = torch.tensor([x1, y1])
    M = torch.tensor([[float(ow)/bw, 0],  # scaling linear transform
                      [0, float(oh)/bh]], dtype=torch.float32)
    dst_keypoints = torch.zeros(keypoints.shape, dtype=torch.int32)
    for i in range(keypoints.shape[0]):
        if keypoints[i,2] == 0:   # visibility
            continue
        dst_vec = torch.matmul(M, (keypoints[i,:2] - crop_origin).float()).int()
        if 0 <= dst_vec[0] < ow and 0 <= dst_vec[1] < oh:
            dst_keypoints[i,:2] = dst_vec
            dst_keypoints[i,2] = keypoints[i,2]
        else:
            # keypoints outside of the image size
            dst_keypoints[i] = torch.tensor([0, 0, 0])

    return image, dst_keypoints


def crop_image(image, keypoints, scale, center,
               box_width, box_height, out_image_size):
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
    logger.debug(f"x1({x1}), y1({y1}), bw({bw}), bh({bh})")

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

    # Transform keypoints
    crop_origin = torch.tensor([x1, y1])
    M = torch.tensor([[float(ow)/bw, 0],  # scaling linear transform
                      [0, float(oh)/bh]], dtype=torch.float32)
    dst_keypoints = torch.zeros(keypoints.shape, dtype=torch.int32)
    for i in range(keypoints.shape[0]):
        if keypoints[i,2] == 0:   # visibility
            continue
        dst_vec = torch.matmul(M, (keypoints[i,:2] - crop_origin).float()).int()
        if 0 <= dst_vec[0] < ow and 0 <= dst_vec[1] < oh:
            dst_keypoints[i,:2] = dst_vec
            dst_keypoints[i,2] = keypoints[i,2]
        else:
            # keypoints outside of the image size
            dst_keypoints[i] = torch.tensor([0, 0, 0])

    return image, dst_keypoints
