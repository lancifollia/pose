#!/usr/bin/env python3
import pickle
import time
from collections import OrderedDict
from typing import Any

import torch
from loguru import logger
from torch import optim, nn
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from pose.data.dataset import CocoDataset
from pose.data.transforms import transform_bbox_pred_to_image_pred


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


def save_debug_info(debug_info: Any,
                    keyword: str,
                    epoch: int,
                    n_iter: int
                    ) -> None:
    fpath = f"debug/pose_{keyword}_epoch{epoch:03}_iter{n_iter:03}.pkl"
    with open(fpath, "wb") as fout:
        pickle.dump(debug_info, fout, pickle.HIGHEST_PROTOCOL)
    logger.info(f"Debugging data saved: {fpath}")


def train_simcc(cfg: CfgNode,
                tr_dataloader: DataLoader,
                model: nn.Module,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                epoch: int,
                is_debug: bool = False
                ) -> float:

    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    start_ts = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(tr_dataloader):
        # measure data loading time
        data_time.update(time.time() - start_ts)

        if type(input) is not torch.Tensor:
            input = torch.tensor(input, dtype=torch.float32)
        input = input.cuda(non_blocking=True)
        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True).float()

        output_x, output_y = model(input)
        loss = criterion(output_x, output_y, target_x, target_y, target_weight)

        if is_debug:
            d = {"input": input, "meta": meta,
                 "target_x": target_x.cpu(), "target_y": target_y.cpu(),
                 "target_weight": target_weight.cpu(),
                 "output_x": output_x.cpu(), "output_y": output_y.cpu(),
                 "loss": loss.item()
                }
            save_debug_info(d, "batch_data", epoch, i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - start_ts)
        start_ts = time.time()

        if i % cfg.TRAIN.LOGGING_FREQ == 0 or i == len(tr_dataloader) - 1:
            msg = f"Train: [{i}/{len(tr_dataloader) - 1}] " \
                  f"Loss {losses.val:.6f}({losses.avg:.6f}) "\
                  f"Speed {input.shape[0] / batch_time.val:.2f} samples/sec"
            logger.info(msg)

        if is_debug and i >= cfg.TRAIN.DEBUG_ITER:
            logger.warning(f"Stop training (iteration exceeds DEBUG_ITER(={cfg.TRAIN.DEBUG_ITER}))")
            break

    return losses.avg


def validate_simcc(cfg: CfgNode,
                   val_dataloader: DataLoader,
                   val_dataset: CocoDataset,
                   model: nn.Module,
                   criterion: nn.Module,
                   epoch: int,
                   is_debug: bool = False
                   ) -> tuple[OrderedDict[str,float], float]:

    batch_time = AverageMeter()
    losses = AverageMeter()

    pred_bbox_size_wh = cfg.MODEL.INPUT_SIZE_WH
    num_kp = cfg.DATA_COMMON.NUM_KEYPOINTS
    simcc_split_factor = cfg.MODEL.EXTRA.SIMCC_SPLIT_FACTOR

    # switch to evaluation mode
    model.eval()
    preds_per_image: dict[int, list[torch.Tensor]] = {}

    with torch.no_grad():
        start_ts = time.time()
        for i, (input, target_x, target_y, target_weight, meta) in enumerate(val_dataloader):
            if type(input) is not torch.Tensor:
                input = torch.tensor(input, dtype=torch.float32)
            input = input.cuda(non_blocking=True)
            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            output_x, output_y = model(input)
            loss = criterion(output_x, output_y, target_x, target_y, target_weight)

            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - start_ts)
            start_ts = time.time()

            output_x = nn.functional.softmax(output_x, dim=2)  # (b, k, logits)
            output_y = nn.functional.softmax(output_y, dim=2)  # (b, k, logits)

            # TODO: need to implement confidence based cutoff
            _, preds_x = output_x.max(2, keepdim=True)
            _, preds_y = output_y.max(2, keepdim=True)

            kps = torch.zeros([input.size(0), num_kp, 2], dtype=torch.float32)  # 2: (x,y)
            kps[:,:,0] = torch.squeeze(torch.true_divide(preds_x, simcc_split_factor))
            kps[:,:,1] = torch.squeeze(torch.true_divide(preds_y, simcc_split_factor))
            bbox_centers = meta["bbox_center"]
            bbox_scales = meta["bbox_scale"]
            image_ids = meta["image_id"]
            for b_idx in range(input.size(0)):
                image_id = int(image_ids[b_idx])
                preds_per_image.setdefault(image_id, [])
                final_pred = transform_bbox_pred_to_image_pred(kps[b_idx],
                                                               bbox_centers[b_idx].numpy(),
                                                               bbox_scales[b_idx].numpy(),
                                                               pred_bbox_size_wh)
                preds_per_image[image_id].append(final_pred)

            if i % cfg.VALIDATE.LOGGING_FREQ == 0 or i == len(val_dataloader) - 1:
                msg = f"Validation: [{i}/{len(val_dataloader) - 1}] " \
                      f"Loss {losses.val:.6f}({losses.avg:.6f})"
                logger.info(msg)

        logger.info(f"AP evaluation target images: {len(preds_per_image):,}")
        num_target_annotations = sum([len(preds) for preds in preds_per_image.values()])
        logger.info(f"AP evaluation target annotations: {num_target_annotations:,}")

        eval_stats = val_dataset.evaluate(preds_per_image, epoch)

        logger.info(", ".join(f"{_m}: {eval_stats[_m]:.4f}"
                              for _m in ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)']))
        logger.info(", ".join(f"{_m}: {eval_stats[_m]:.4f}"
                              for _m in ['AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']))

        if is_debug:
            d = {"val_loss_mean": losses.avg,
                 "eval_stats": eval_stats,
                 "preds_per_image": preds_per_image
                 }
            save_debug_info(d, "validation", epoch, 0)

        return eval_stats, losses.avg
