#!/usr/bin/env python3
import os
import sys

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pose.config import get_cfg_defaults
from pose.data.dataset import CocoDataset


logger.remove()
logger.add(sys.stderr, level="INFO")


cfg = get_cfg_defaults()
dataset = CocoDataset(cfg, is_train=True, transform=None)
logger.info(f"Data size: {len(dataset):,}")

start_idx = 0
for i in tqdm(range(start_idx, len(dataset))):
    try:
        dataset[i]
    except RuntimeError as e:
        logger.info(f"Error occurs at data idx {i}")
        logger.info(f"Error message: {e}")
        break


logger.info("Done")
