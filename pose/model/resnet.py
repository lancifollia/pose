#!/usr/bin/env python3

from typing import List, Optional, Type, Union

import torch
from einops import rearrange
from loguru import logger
from torch import nn, Tensor
from yacs.config import CfgNode


BN_MOMENTUM = 0.1


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 ) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 cfg: CfgNode,
                 ) -> None:
        super(PoseResNet, self).__init__()
        self.num_kp = cfg.DATA_COMMON.NUM_KEYPOINTS
        self.num_channel_per_kp = cfg.MODEL.EXTRA.NUM_CHANNEL_PER_KP
        self.final_conv_kernel_size = cfg.MODEL.EXTRA.FINAL_CONV_KERNEL_SIZE
        self.resolution_wh = Tensor(cfg.MODEL.INPUT_SIZE_WH).type(torch.int32)
        self.cc_size_wh = self.resolution_wh * int(cfg.MODEL.EXTRA.SIMCC_SPLIT_FACTOR)
        logger.info(f"coordinate classification size: {self.cc_size_wh}")

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.final_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=self.num_kp * self.num_channel_per_kp,
            kernel_size=self.final_conv_kernel_size,
            stride=1,
            padding=1 if self.final_conv_kernel_size == 3 else 0
        )

        total_stride = 2 ** 5
        final_layer_resolution_wh = (self.resolution_wh / total_stride).type(torch.int32)
        fw, fh = map(int, final_layer_resolution_wh)
        logger.info(f"final layer resolution: w({fw}), h({fh})")
        head_in_size = int(fw * fh * self.num_channel_per_kp)
        logger.info(f"head-in size: {head_in_size}")
        self.cc_head_x = nn.Linear(head_in_size, self.cc_size_wh[0])
        self.cc_head_y = nn.Linear(head_in_size, self.cc_size_wh[1])
        logger.info(f"head x cc size: {self.cc_size_wh[0]}")
        logger.info(f"head y cc shape: {self.cc_size_wh[1]}")

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1
                    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_layer(x)
        # b: batch size
        x = rearrange(x, 'b (k c) h w -> b k (c h w)',
                      k=self.num_kp, c=self.num_channel_per_kp)

        pred_x = self.cc_head_x(x)
        pred_y = self.cc_head_y(x)

        return pred_x, pred_y

    def init_weights(self) -> None:
        logger.info("Init weights from normal distribution.")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


RESNET_SPEC: dict[int, tuple[Type[Union[BasicBlock, Bottleneck]], List[int]]] = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_model(cfg: CfgNode, need_init_weights: bool = False) -> nn.Module:
    block, layers = RESNET_SPEC[cfg.MODEL.EXTRA.NUM_RESNET_LAYERS]
    model = PoseResNet(block, layers, cfg)
    if need_init_weights:
        model.init_weights()
    return model
