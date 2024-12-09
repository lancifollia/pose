#!/usr/bin/env python3

"""
Training script
"""

from loguru import logger

import _init_paths
from data.dataset import CocoDataset


def train():
    # Setup configuration
    cfg = {}

    # Prepare dataset
    tr_dataset = CocoDataset(cfg['anno_fname'],
                             cfg['image_dir'])
    val_dataset = CocoDataset(cfg['anno_fname'],
                              cfg['image_dir'])

    # Prepare model parameter

    # Iterate training epochs

    logger.info('Model training is complete')


if __name__ == '__main__':
    train()
