#!/usr/bin/env python3

import torch
from torch import nn, Tensor


class KLDiscretLoss(nn.Module):
    def __init__(self) -> None:
        super(KLDiscretLoss, self).__init__()
        # TODO: Is it right?
        # self.log_softmax = nn.LogSoftmax(dim=1)   # (b, logits)
        self.softmax = nn.Softmax(dim=1)   # (b, logits)
        self.kl_div_loss= nn.KLDivLoss(reduction='none')


    def criterion(self, output: Tensor, target: Tensor) -> Tensor:
        # TODO: Is it right?
        # scores = self.log_softmax(output)
        scores = self.softmax(output)
        loss = torch.mean(self.kl_div_loss(scores, target), dim=1)
        return loss

    def forward(self,
                output_y: Tensor, output_x: Tensor,
                target_y: Tensor, target_x: Tensor,
                target_weight: Tensor
                ) -> Tensor:
        num_kp = output_y.shape[1]  # (b, k, (c h w))
        loss = torch.tensor(0.).cuda()

        for idx in range(num_kp):
            _output_y = output_y[:,idx].squeeze()
            _output_x = output_x[:,idx].squeeze()
            _target_y = target_y[:,idx].squeeze()
            _target_x = target_x[:,idx].squeeze()
            weight = target_weight[:,idx].squeeze()
            loss += (self.criterion(_output_y, _target_y).mul(weight).mean())
            loss += (self.criterion(_output_x, _target_x).mul(weight).mean())
        return loss / num_kp
