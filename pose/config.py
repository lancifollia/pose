#!/usr/bin/env python3
from yacs.config import CfgNode as CN


_C = CN()

# CfgNodes should be alphabetical ordered.


"""Training Data"""
_C.DATA_TRAIN = CN()
_C.DATA_TRAIN.ANNOTATION_FNAME = "/home/dev/data/coco/annotations/person_keypoints_train2017.json"
_C.DATA_TRAIN.IMAGE_DIR = "/home/dev/data/coco/images/train2017/"
# Minimum required keypoints for training sample.
_C.DATA_TRAIN.MIN_KEYPOINTS = 1


"""Data augmentation"""
_C.DATA_TRAIN.DATA_AUG = CN()
# Horizontal flipping trigering probability
_C.DATA_TRAIN.DATA_AUG.FLIP_PROBA = 0.5
# Image rotating factor: angle in degrees (-180, 180)
_C.DATA_TRAIN.DATA_AUG.ROTATE_FACTOR = 40
# Image rescaling factor: ratio (0.0:do not scaling, 1.0:upto 200% scaling)
_C.DATA_TRAIN.DATA_AUG.SCALE_FACTOR = 0.3
# Half body transforming trigering probability
_C.DATA_TRAIN.DATA_AUG.HALF_BODY_PROBA = 0.3
# Half body transforming trigering minimum number of keypoints
_C.DATA_TRAIN.DATA_AUG.HALF_BODY_MIN_KEYPOINTS = 8


"""Neural network settings"""
_C.NN = CN()
# Neural network input image size. [height, width]
_C.NN.INPUT_SIZE = [256, 192]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
