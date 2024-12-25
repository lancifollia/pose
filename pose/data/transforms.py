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



def crop_image(image, keypoints, scale, center,
               box_width, box_height, out_image_size):
    ih, iw = image.shape[1:]  # input image shape
    oh, ow = out_image_size
    aspect_ratio = float(ow) / oh

    # Resizing bbox & cropping
    logger.info(f"center: {center}")
    bw = int(min(box_width * scale, 2 * (iw - center[0]) , 2 * center[0]))
    logger.info(f"bw, {box_width * scale}, {2 * (iw - center[0])}, {2 * center[0]}")
    bh = int(min(box_height * scale, 2 * (ih - center[1]), 2 * center[1]))
    logger.info(f"bh, {box_height * scale}, {2 * (ih - center[1])}, {2 * center[1]}")
    if bw >= bh * aspect_ratio:
        bw = int(bh * aspect_ratio)
    else:
        bh = int(float(bw) * aspect_ratio**-1)
    x1 = center[0] - math.ceil(bw * 0.5)
    y1 = center[1] - math.ceil(bh * 0.5)
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
