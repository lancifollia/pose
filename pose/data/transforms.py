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
    if box_width >= box_height * aspect_ratio:
        bw = box_width * scale
        bh = int(float(bw) / aspect_ratio)
    else:
        bh = int(box_height * scale)
        bw = int(bh * aspect_ratio)
    x1 = max(center[0] - math.ceil(bw * 0.5), 0)
    x2 = min(center[0] + math.floor(bw * 0.5), iw-1)
    y1 = max(center[1] - math.ceil(bh * 0.5), 0)
    y2 = min(center[1] + math.floor(bh * 0.5), ih-1)
    image = image[:,y1:y2,x1:x2]

    # Resizing cropped image
    image = resize(image, size=out_image_size,
                   interpolation=InterpolationMode.BILINEAR)

    # Transform keypoints
    ic = torch.tensor(center)                          # center of input
    oc = torch.tensor([ow/2, oh/2], dtype=torch.int32) # center of output
    M = torch.tensor([[ow/float(bw), 0], [0, oh/float(bh)]], dtype=torch.float32)
    dst_keypoints = torch.zeros(keypoints.shape, dtype=torch.int32)
    for i in range(keypoints.shape[0]):
        if keypoints[i,2] == 0:   # visibility
            continue
        dst_vec = torch.matmul(M, (keypoints[i,:2] - ic).float()).int() + oc
        if 0 <= dst_vec[0] < ow and 0 <= dst_vec[1] < oh:
            dst_keypoints[i,:2] = dst_vec
            dst_keypoints[i,2] = keypoints[i,2]
        else:
            # keypoints outside of the image size
            dst_keypoints[i] = torch.tensor([0, 0, 0])

    return image, dst_keypoints
