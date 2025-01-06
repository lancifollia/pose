#!/usr/bin/env python3

from torch import nn
from yacs.config import CfgNode


BN_MOMENTUM = 0.1


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1
            ) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):
    def __init__(self,
                 block,
                 layers
                 ) -> None:
        super(PoseResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # TODO: final layer & y,x classification layer for each

    def forward(self, x):
        # TODO
        return None

    def _make_layer(self, block, planes, blocks, stride=1):
        # TODO
        return None


RESNET_SPEC = {
    18: (BasicBlock, [2, 2, 2, 2]),
    # 34: (BasicBlock, [3, 4, 6, 3]),
    # 50: (Bottleneck, [3, 4, 6, 3]),   -> TODO: Bottleneck
    # 101: (Bottleneck, [3, 4, 23, 3]),
    # 152: (Bottleneck, [3, 8, 36, 3])
}

def get_model(cfg: CfgNode) -> nn.Module:
    block, layers = RESNET_SPEC[cfg.MODEL.EXTRA.NUM_RESNET_LAYERS]
    model = PoseResNet(block, layers)
    return model
