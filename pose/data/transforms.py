#!/usr/bin/env python3

import torch


class RescaleImgDiv255(object):

    def __call__(self, pic):
        if type(pic) not in [torch.tensor, torch.Tensor]:
            raise ValueError("Only torch.tensor type is supported.")
        if pic.dtype != torch.uint8:
            raise ValueError("dtype should be torch.uint8")
        return pic.div(255)
