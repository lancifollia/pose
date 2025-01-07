"""
Training script
"""
import os
import sys

from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
from yacs.config import CfgNode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pose.config import get_cfg_defaults
from pose.data.dataset import CocoDataset
from pose.data.transforms import RescaleImgDiv255


logger.remove()
logger.add(sys.stderr, level="INFO")


def get_model(cfg: CfgNode) -> nn.Module:
    if cfg.MODEL.NAME == "resnet":
        from pose.model.resnet import get_model
        return get_model(cfg)
    else:
        raise ValueError(f"Invalid model name: {cfg.MODEL.NAME}")


def train(config_fname: str) -> None:
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_fname)
    cfg.freeze()
    logger.info(f"Configurations:\n{cfg}")

    input_transform = Compose([
        RescaleImgDiv255(),
        Normalize(mean=[0.485, 0.456, 0.406], # imagenet data normalization
                  std=[0.229, 0.224, 0.225])
    ])

    logger.info("Preapring training dataset")
    tr_dataset = CocoDataset(cfg, is_train_data=True,
                             transform=input_transform)
    logger.info("Preparing validation dataset")
    val_dataset = CocoDataset(cfg, is_train_data=False,
                              transform=input_transform)

    logger.info("Preapring dataloader")
    tr_dataloader = DataLoader(tr_dataset,
                               batch_size=cfg.TRAINING.BATCH_SIZE,
                               shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.VALIDATION.BATCH_SIZE,
                                shuffle=True)

    # Prepare model parameter
    model = get_model(cfg)
    print(f"Model: {model}")

    # Iterate training epochs

    logger.info('Model training is complete')



if __name__ == '__main__':
    train(sys.argv[1])
