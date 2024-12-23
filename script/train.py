#!/usr/bin/env python3

"""
Training script
"""
import sys
from os import path

import yaml
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

sys.path.insert(0, path.join(path.dirname(__file__), '..', 'pose'))

from data.dataset import CocoDataset
from data.transforms import RescaleImgDiv255


def train(config_fname):
    with open(config_fname, encoding="utf-8") as config_fin:
        cfg = yaml.load(config_fin, Loader=yaml.Loader)
    logger.info(f"Config dump: {cfg}")

    input_transform = Compose([
        RescaleImgDiv255(),
        Normalize(mean=[0.485, 0.456, 0.406], # imagenet data normalization
                  std=[0.229, 0.224, 0.225])
    ])

    logger.info("Preapring training dataset")
    tr_dataset = CocoDataset(cfg['train_data']['anno_fname'],
                             cfg['train_data']['image_dir'],
                             transform=input_transform)
    logger.info("Loading validation dataset")
    val_dataset = CocoDataset(cfg['val_data']['anno_fname'],
                              cfg['val_data']['image_dir'],
                              transform=input_transform)

    logger.info("Preapring dataloader")
    tr_dataloader = DataLoader(tr_dataset,
                               batch_size=cfg['training']['batch_size'],
                               shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg['training']['batch_size'],
                                shuffle=True)

    # Prepare model parameter
    device = "cuda"
    model = None

    # Iterate training epochs

    logger.info('Model training is complete')


if __name__ == '__main__':
    train(sys.argv[1])
