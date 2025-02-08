"""
Training script
"""
import datetime
import os
import pathlib
import sys
import time

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
from yacs.config import CfgNode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pose.config import get_cfg_defaults
from pose.core.function import train_simcc, validate_simcc
from pose.core.loss import KLDiscretLoss
from pose.data.dataset import CocoDataset
from pose.data.transforms import RescaleImgDiv255


logger.remove()
logger.add(sys.stderr, level="INFO")


def get_loss_function(loss_type: str) -> nn.Module:
    if loss_type == "KLDiscretLoss":
        return KLDiscretLoss()
    else:
        msg = "Only the loss_type 'KLDiscrteLoss' is supported yet. " \
              f"Invalid loss type: {loss_type}"
        raise RuntimeError(msg)


def get_model(cfg: CfgNode, need_init_weights: bool) -> nn.Module:
    if cfg.MODEL.NAME == "resnet":
        from pose.model.resnet import get_model
        return get_model(cfg, need_init_weights)
    else:
        msg = "Only the model 'resnet' is supported yet. " \
              f"Invalid model name: {cfg.MODEL.NAME}"
        raise RuntimeError(msg)


def train(config_fname: str) -> None:
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_fname)
    cfg.freeze()
    logger.info(f"Configurations:\n{cfg}")

    torch.cuda.set_device("cuda:0")

    # Prepare model parameter
    model = get_model(cfg, need_init_weights=True).cuda()
    logger.info(f"Model: {model}")
    total_param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model param size: {total_param_size:,}")

    # Set loss function
    criterion = get_loss_function(cfg.MODEL.LOSS_TYPE).cuda()
    logger.info(f"Loss type:\n{criterion}")

    # Data loading
    input_transform = Compose([
        RescaleImgDiv255(),
        Normalize(mean=[0.406, 0.485, 0.456], # imagenet data normalization (RGB order)
                  std=[0.225, 0.229, 0.224])
    ])
    logger.info("Preapring training dataset")
    tr_dataset = CocoDataset(cfg, is_train=True,
                             transform=input_transform)
    logger.info("Preparing validation dataset")
    val_dataset = CocoDataset(cfg, is_train=False,
                              transform=input_transform)
    logger.info("Preapring dataloader")
    tr_dataloader = DataLoader(tr_dataset,
                               batch_size=cfg.DATA_LOADER.BATCH_SIZE,
                               shuffle=True,
                               num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                               pin_memory=cfg.DATA_LOADER.USE_PIN_MEMORY,
                               )
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.DATA_LOADER.BATCH_SIZE,
                                shuffle=False,
                                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                pin_memory=cfg.DATA_LOADER.USE_PIN_MEMORY,
                                )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    logger.info(f"Optimizer:\n{optimizer}")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=-1
    )

    logging_fname = f"perf_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.log"
    logger.info(f"performance logging file name: {logging_fname}")
    with open(logging_fname, 'w') as fout:
        fout.write("| ep | AP | AR | Train | Val |\n")
        fout.write("| --- | --- | --- | --- | --- |\n")

    checkpoint_dir = "checkpoints"
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    begin_epoch = 0
    best_ap = -1.0

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        start_ts = time.time()
        logger.info(f"{'=-'*5} Epoch[{epoch}] started {'-='*5}")

        train_loss = train_simcc(cfg, tr_dataloader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        ap, ar, val_loss = validate_simcc(cfg, val_dataloader, val_dataset, model, criterion, epoch)

        with open(logging_fname, 'a') as fout:
            fout.write(f"| {epoch} | {ap:.4f} | {ar:.4f} | {train_loss:.6f} | {val_loss:.6f} |\n")

        if ap > best_ap:
            best_ap = ap
            logger.info(f"Best AP is updated! {ap:.4f}")

        checkpoint_fpath = os.path.join(checkpoint_dir, "checkpoint_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': {'train': train_loss, 'val': val_loss},
            'avg_precision': ap,
            'best_avg_precision': best_ap,
        }, checkpoint_fpath)

        logger.info(f"Epoch[{epoch}] complete. Elapsed at {(time.time() - start_ts)/60:.2f} mins")

    logger.info('Model training is complete')



if __name__ == '__main__':
    train(sys.argv[1])
