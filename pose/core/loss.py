#!/usr/bin/env python3

import torch
from torch import nn, Tensor


class KLDiscretLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super(KLDiscretLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)   # (b, logits)
        self.kl_div_loss= nn.KLDivLoss(reduction="none", log_target=False)
        if reduction not in ["mean", "weighted_mean"]:
            msg = "expected reduction to be one of 'mean', or 'weighted_mean' "\
                  f"but found '{reduction}'"
            raise RuntimeError(msg)
        else:
            self.reduction = reduction

    def criterion(self, output: Tensor, target: Tensor) -> Tensor:
        """
        The target tensor is converted to log scale inside the kl_div_loss() method.
        See more: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        """
        scores = self.log_softmax(output)
        loss = torch.mean(self.kl_div_loss(scores, target), dim=1)
        return loss

    def forward(self,
                output_x: Tensor,
                output_y: Tensor,
                target_x: Tensor,
                target_y: Tensor,
                target_weight: Tensor
                ) -> Tensor:
        num_kp = output_y.shape[1]  # (b, k, (c h w))
        loss = torch.tensor(0.).cuda()

        for idx in range(num_kp):
            _output_x = output_x[:,idx]
            _output_y = output_y[:,idx]
            _target_x = target_x[:,idx]
            _target_y = target_y[:,idx]
            weight = target_weight[:,idx].squeeze()  # (b, k, 1) -> (b, k)

            if self.reduction == "mean":
                loss += self.criterion(_output_x, _target_x).mul(weight).mean()
                loss += self.criterion(_output_y, _target_y).mul(weight).mean()
            elif self.reduction == "weighted_mean":
                loss += self.criterion(_output_x, _target_x).mul(weight).sum() * weight.sum()**-1
                loss += self.criterion(_output_y, _target_y).mul(weight).sum() * weight.sum()**-1
            else:
                raise RuntimeError(f"Invalid reduction type: {self.reduction}")

        return loss / num_kp
