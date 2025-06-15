#!/usr/bin/env python3

import torch
from torch import nn, Tensor


class KLDiscretLoss(nn.Module):
    def __init__(self, device: str) -> None:
        super(KLDiscretLoss, self).__init__()
        self.device = device
        self.log_softmax = nn.LogSoftmax(dim=1).to(self.device)   # (b, logits)
        self.kl_div_loss= nn.KLDivLoss(reduction="none", log_target=False).to(self.device)

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
        loss = torch.tensor(0.).to(self.device)

        for idx in range(num_kp):
            _pred_x = pred_x[:,idx]
            _pred_y = pred_y[:,idx]
            _target_x = target_x[:,idx]
            _target_y = target_y[:,idx]
            weight = target_weight[:,idx].squeeze()  # (b, k, 1) -> (b, k)
            loss += (self.criterion(_pred_x, _target_x).mul(weight).mean())
            loss += (self.criterion(_pred_y, _target_y).mul(weight).mean())
        return loss / num_kp
