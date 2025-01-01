#!/usr/bin/env python3

"""
Training script
"""
import sys
from os import path

from loguru import logger
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

sys.path.insert(0, path.join(path.dirname(__file__), '..', 'pose'))

from config import get_cfg_defaults
from data.dataset import CocoDataset
from data.transforms import RescaleImgDiv255


def train(config_fname):
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
    tr_dataset = CocoDataset(anno_fname=cfg.DATA_TRAIN.ANNOTATION_FNAME,
                             image_dir=cfg.DATA_TRAIN.IMAGE_DIR,
                             nn_input_size=cfg.NN.INPUT_SIZE,
                             data_aug_cfg=cfg.DATA_TRAIN.DATA_AUG,
                             min_keypoints=cfg.DATA_COMMON.MIN_KEYPOINTS,
                             transform=input_transform)
    logger.info("Loading validation dataset"  )
    val_dataset = CocoDataset(anno_fname=cfg.DATA_VAL.ANNOTATION_FNAME,
                              image_dir=cfg.DATA_VAL.IMAGE_DIR,
                              nn_input_size=cfg.NN.INPUT_SIZE,
                              data_aug_cfg=cfg.DATA_VAL.DATA_AUG,
                              min_keypoints=cfg.DATA_COMMON.MIN_KEYPOINTS,
                              transform=input_transform)

    logger.info("Preapring dataloader")
    tr_dataloader = DataLoader(tr_dataset,
                               batch_size=cfg.TRAINING.BATCH_SIZE,
                               shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.VALIDATION.BATCH_SIZE,
                                shuffle=True)

    # Prepare model parameter
    device = "cuda"
    model = None

    # Iterate training epochs

    logger.info('Model training is complete')


if __name__ == '__main__':
    train(sys.argv[1])
