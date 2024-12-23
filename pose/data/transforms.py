#!/usr/bin/env python3
import math

import torch

from torchvision.transforms.v2.functional import affine
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
