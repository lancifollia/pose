#!/usr/bin/env python3
import time

import torch
from loguru import logger
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from yacs.config import CfgNode


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.cum_val = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.cum_val += val * n
        self.count += n
        self.avg = self.cum_val / self.count if self.count != 0 else 0


def train_simcc(cfg: CfgNode,
                tr_dataloader: DataLoader,
                model: nn.Module,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                epoch: int
                ) -> None:

    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    start_ts = time.time()
    for i, (input, target_y, target_x, target_weight, meta) in enumerate(tr_dataloader):
        # measure data loading time
        data_time.update(time.time() - start_ts)

        input = input.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        target_x = target_x.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True).float()

        output_y, output_x = model(input)

        loss = criterion(output_y, output_x, target_y, target_x, target_weight)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), input.shape[0])
        batch_time.update(time.time() - start_ts)
        start_ts = time.time()

        if i % cfg.TRAIN.LOGGING_FREQ == 0 or i == len(tr_dataloader) - 1:
            msg = f"Train: [{i}/{len(tr_dataloader) - 1}] " \
                  f"Loss {losses.val:.4e}({losses.avg:.4e})  "\
                  f"Time {batch_time.val:.2f}({batch_time.avg:.2f})  " \
                  f"Speed {input.shape[0] / batch_time.val:.2f} samples/sec  " \
                  f"Data {data_time.val:.2f}({data_time.avg:.2f})"
            logger.info(msg)


def validate_simcc(cfg: CfgNode,
                   val_dataloader: DataLoader,
                   val_dataset: Dataset,
                   model: nn.Module,
                   criterion: nn.Module,
                   epoch: int
                   ) -> None:

    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        start_ts = time.time()
        for i, (input, target_y, target_x, target_weight, meta) in enumerate(val_dataloader):
            input = input.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)
            target_x = target_x.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            output_y, output_x = model(input)
            output_y = nn.functional.softmax(output_y, dim=2)  # (b, k, logits)
            output_x = nn.functional.softmax(output_x, dim=2)  # (b, k, logits)

            loss = criterion(output_y, output_x, target_y, target_x, target_weight)

            losses.update(loss.item(), input.shape[0])

            batch_time.update(time.time() - start_ts)
            start_ts = time.time()

            # TODO: Calculate average precision by COCO dataset evaluation metric.

            if i % cfg.VALIDATE.LOGGING_FREQ == 0 or i == len(val_dataloader) - 1:
                msg = f"Validation: [{i}/{len(val_dataloader) - 1}] " \
                      f"Loss {losses.val:.4e}({losses.avg:.4e})  "\
                      f"Time {batch_time.val:.2f}({batch_time.avg:.2f})"
                logger.info(msg)
