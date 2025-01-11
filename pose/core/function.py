#!/usr/bin/env python3
import torch
from loguru import logger
from torch import optim, nn
from torch.utils.data import DataLoader
from yacs.config import CfgNode


def train_simcc(cfg: CfgNode,
                tr_dataloader: DataLoader,
                model: nn.Module,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                epoch: int
                ) -> None:

    # switch to train mode
    model.train()

    for i, (input, target_y, target_x, target_weight, extra_info) in enumerate(tr_dataloader):
        input = input.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        target_x = target_x.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True).float()

        output_y, output_x = model(input)

        loss = criterion(output_y, output_x, target_y, target_x, target_weight)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % cfg.TRAIN.LOGGING_FREQ == 0:
            logloss = torch.log(loss)
            msg = f"Epoch[{epoch}] Batch[{i}/{len(tr_dataloader)}] logloss {logloss:.5f}"
            logger.info(msg)
