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

    def criterion(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        The target tensor is converted to log scale inside the kl_div_loss() method.
        See more: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        """
        scores = self.log_softmax(pred)
        loss = torch.mean(self.kl_div_loss(scores, target), dim=1)
        return loss

    def forward(self,
                pred_x: Tensor,
                pred_y: Tensor,
                target_x: Tensor,
                target_y: Tensor,
                target_weight: Tensor
                ) -> Tensor:
        num_kp = pred_x.shape[1]  # (b, k, (c h w))
        loss = torch.tensor(0.).cuda()

        for idx in range(num_kp):
            _pred_x = pred_x[:,idx].squeeze()
            _pred_y = pred_y[:,idx].squeeze()
            _target_x = target_x[:,idx].squeeze()
            _target_y = target_y[:,idx].squeeze()
            weight = target_weight[:,idx].squeeze()  # (b, k, 1) -> (b, k)

            if self.reduction == "mean":
                loss_x = (self.criterion(_pred_x, _target_x).mul(weight).mean())
                loss_y = (self.criterion(_pred_y, _target_y).mul(weight).mean())
                loss = loss + loss_x + loss_y
            elif self.reduction == "weighted_mean":
                loss += (self.criterion(_pred_x, _target_x).mul(weight).sum() * weight.sum()**-1)
                loss += (self.criterion(_pred_y, _target_y).mul(weight).sum() * weight.sum()**-1)
            else:
                raise RuntimeError(f"Invalid reduction type: {self.reduction}")

        return loss / num_kp
