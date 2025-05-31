"""
Training script
"""
import datetime
import os
import pathlib
import pickle
import random
import sys
import time

import torch
import numpy as np
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Normalize
from yacs.config import CfgNode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pose.config import get_cfg_defaults
from pose.core.function import train_simcc, validate_simcc
from pose.core.loss import KLDiscretLoss
from pose.data.dataset import CocoDataset


logger.remove()
logger.add(sys.stderr, level="INFO")


def get_loss_function(loss_type: str) -> nn.Module:
    if loss_type == "KLDiscretLoss":
        return KLDiscretLoss()
    else:
        msg = "Only the loss_type 'KLDiscretLoss' is supported yet. " \
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


def set_random_seed(random_seed: int) -> None:
    logger.info(f"Set random seed: {random_seed}")
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def dump_parameters(model: nn.Module, path: str) -> None:
    params = []
    for p in model.parameters():
        params.append(p)
    with open(path, "wb") as fout:
        pickle.dump(params, fout, pickle.HIGHEST_PROTOCOL)


def train(config_fname: str, mode: str) -> None:
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_fname)
    cfg.freeze()
    logger.info(f"Configurations:\n{cfg}")

    torch.cuda.set_device("cuda:0")

    # Prepare model parameter
    set_random_seed(42)
    model = get_model(cfg, need_init_weights=True).cuda()

    pathlib.Path("debug/").mkdir(parents=True, exist_ok=True)

    if cfg.TRAIN.DEBUG:
        fpath = 'debug/pose_init_model.pt'
        dump_parameters(model, fpath)
        logger.warning(f"Initial model parameter dumped: {fpath}")

    # logger.info(f"Model: {model}")
    total_param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model param size: {total_param_size:,}")

    # Set loss function
    criterion = get_loss_function(cfg.MODEL.LOSS_TYPE).cuda()
    logger.info(f"Loss type:\n{criterion}")

    # Data loading
    input_transform = Compose([
        transforms.ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], # imagenet data normalization parameter
                  std=[0.229, 0.224, 0.225])
    ])
    logger.info("Preapring training dataset")
    tr_dataset = CocoDataset(cfg, is_train=True, is_debug=cfg.TRAIN.DEBUG,
                             transform=input_transform)
    logger.info("Preparing validation dataset")
    val_dataset = CocoDataset(cfg, is_train=False, is_debug=cfg.TRAIN.DEBUG,
                              transform=input_transform)
    logger.info("Preapring dataloader")
    tr_dataloader = DataLoader(tr_dataset,
                               batch_size=cfg.DATA_LOADER.BATCH_SIZE,
                               shuffle=cfg.DATA_LOADER.TRAIN_DATA_SHUFFLE,
                               num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                               pin_memory=cfg.DATA_LOADER.USE_PIN_MEMORY,
                               )
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.DATA_LOADER.BATCH_SIZE,
                                shuffle=False,
                                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                pin_memory=cfg.DATA_LOADER.USE_PIN_MEMORY,
                                )

    if mode == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
        logger.info(f"Optimizer:\n{optimizer}")
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=-1
        )

        logging_performance = cfg.TRAIN.LOGGING_EPOCH_PERF
        if logging_performance:
            logging_fname = f"perf_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.log"
            logger.info(f"performance logging file name: {logging_fname}")
            with open(logging_fname, 'w') as fout:
                fout.write("| ep | AP | AP.5 | AP.75 | AR | Train | Val |\n")
                fout.write("| --- | --- | --- | --- | --- | --- | --- |\n")

        if cfg.TRAIN.SAVE_CHECKPOINTS:
            checkpoint_dir = "checkpoints"
            pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        begin_epoch = 0
        best_ap = -1.0
        for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
            start_ts = time.time()
            logger.info(f"{'=-'*5} Epoch[{epoch}] started {'-='*5}")

            train_loss = train_simcc(cfg=cfg,
                                     tr_dataloader=tr_dataloader,
                                     model=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     epoch=epoch,
                                     is_debug=cfg.TRAIN.DEBUG)
            lr_scheduler.step()

            if cfg.TRAIN.DEBUG:
                fpath = f"debug/pose_epoch{epoch:03}_model.pt"
                dump_parameters(model, fpath)
                logger.warning(f"Trained model param dumped: {fpath}")

            eval_stats, val_loss = validate_simcc(cfg=cfg,
                                                  val_dataloader=val_dataloader,
                                                  val_dataset=val_dataset,
                                                  model=model,
                                                  criterion=criterion,
                                                  epoch=epoch,
                                                  is_debug=cfg.TRAIN.DEBUG
                                                  )

            if logging_performance:
                with open(logging_fname, 'a') as fout:
                    _vals = [f"{epoch}"]
                    for _m in ['AP', 'AP .5', 'AP .75', 'AR']:
                        _vals.append(f"{eval_stats[_m]:.4f}")
                    _vals.append(f"{train_loss:.6f}")
                    _vals.append(f"{val_loss:.6f}")
                    fout.write("| " + " | ".join(_vals) + " |\n")

            if eval_stats['AP'] > best_ap:
                best_ap = eval_stats['AP']
                logger.info(f"Best AP is updated! {eval_stats['AP']:.4f}")

            if cfg.TRAIN.SAVE_CHECKPOINTS:
                checkpoint_fpath = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': {'train': train_loss, 'val': val_loss},
                    'avg_precision': eval_stats['AP'],
                    'best_avg_precision': best_ap,
                }, checkpoint_fpath)
                logger.info(f"Save checkpoint parameters at: {checkpoint_fpath}")

            logger.info(f"Epoch[{epoch}] complete. Elapsed at {(time.time() - start_ts)/60:.2f} mins")

            if cfg.TRAIN.DEBUG and epoch == 1:
                logger.info(f"debugging finished. epoch: {epoch}")
                break

        logger.info("Model training complete")
    elif mode == "test":
        logger.info(f"Load model from: {cfg.TEST.MODEL_PATH}")
        model_data = torch.load(cfg.TEST.MODEL_PATH)
        model.load_state_dict(model_data['model_state_dict'])

        eval_stats, val_loss = validate_simcc(cfg=cfg,
                                              val_dataloader=val_dataloader,
                                              val_dataset=val_dataset,
                                              model=model,
                                              criterion=criterion,
                                              epoch=0,
                                              is_debug=cfg.TEST.DEBUG
                                              )
        logger.info("Model testing complete")
    else:
        logger.error(f"Invalid script mode: {mode}")


if __name__ == '__main__':
    mode = sys.argv[1]
    config_fname = sys.argv[2]
    assert mode in ["train", "test"]
    train(config_fname, mode)
