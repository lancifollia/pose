"""
Training script
"""
import os
import sys
import time

from loguru import logger
from torch import cuda, optim, nn
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

    cuda.set_device("cuda:0")

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

    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    logger.info(f"Optimizer:\n{optimizer}")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=-1
    )

    begin_epoch = 0
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        start_ts = time.time()
        logger.info(f"{'=-'*5} Epoch[{epoch}] started {'-='*5}")
        train_simcc(cfg, tr_dataloader, model, criterion, optimizer, epoch)
        validate_simcc(cfg, val_dataloader, val_dataset, model, criterion, epoch)
        lr_scheduler.step()
        logger.info(f"Epoch[{epoch}] complete. Elapsed at {(time.time() - start_ts)/60:.2f} mins")

    logger.info('Model training is complete')



if __name__ == '__main__':
    train(sys.argv[1])
