#!/usr/bin/env python3

"""
Training script
"""
import sys

import yaml
from loguru import logger

import _init_paths
from data.dataset import CocoDataset


def train(config_fname):
    # Setup configuration
    with open(config_fname, encoding="utf-16") as config_fin:
        cfg = yaml.load(config_fin, Loader=yaml.Loader)

    #TODO: Set transform pipeline
    transform = None
    target_transform = None

    # Prepare dataset
    logger.info("Loading training dataset")
    tr_dataset = CocoDataset(cfg['train_data']['anno_fname'],
                             cfg['train_data']['image_dir'],
                             transform=transform,
                             target_transform=target_transform)
    logger.info("Loading validation dataset")
    val_dataset = CocoDataset(cfg['val_data']['anno_fname'],
                              cfg['val_data']['image_dir'],
                              transform=transform,
                              target_transform=target_transform)

    # Prepare model parameter

    # Iterate training epochs

    logger.info('Model training is complete')


if __name__ == '__main__':
    train(sys.argv[1])
