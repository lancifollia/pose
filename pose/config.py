from yacs.config import CfgNode as CN


_C = CN()

# CfgNodes should be alphabetical ordered.


"""Data Common"""
_C.DATA_COMMON = CN()
_C.DATA_COMMON.NUM_KEYPOINTS = 17  # COCO dataset
# Minimum required keypoints for training sample.
_C.DATA_COMMON.MIN_KEYPOINTS = 1



"""Training Data"""
_C.DATA_TRAIN = CN()
_C.DATA_TRAIN.ANNOTATION_FNAME = "/home/dev/data/coco/annotations/person_keypoints_train2017.json"
_C.DATA_TRAIN.IMAGE_DIR = "/home/dev/data/coco/images/train2017/"


"""Training Data >> Data augmentation"""
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


"""Validation Data"""
_C.DATA_VAL = CN()
_C.DATA_VAL.ANNOTATION_FNAME = "/home/dev/data/coco/annotations/person_keypoints_val2017.json"
_C.DATA_VAL.IMAGE_DIR = "/home/dev/data/coco/images/val2017/"
_C.DATA_VAL.DATA_AUG = None


"""Model"""
_C.MODEL = CN()
_C.MODEL.NAME = None
# Neural network input image size. [height, width]
_C.MODEL.INPUT_SIZE = [256, 192]


"""Model.Extra"""
_C.MODEL.EXTRA = CN()

"""ModelExtra > SimCC"""
_C.MODEL.EXTRA.SIMCC_SPLIT_FACTOR = 2
_C.MODEL.EXTRA.SIMCC_SIGMA = 1.0

"""Model.Extra > ResNet"""
# ResNet layer spec: 18, 34, 50, 101, 152
_C.MODEL.EXTRA.NUM_RESNET_LAYERS = 18
_C.MODEL.EXTRA.NUM_CHANNEL_PER_KP = 28
_C.MODEL.EXTRA.FINAL_CONV_KERNEL_SIZE = 1


"""Model training"""
_C.TRAINING = CN()
_C.TRAINING.BATCH_SIZE = 120
_C.TRAINING.NUM_GPU = 1


"""Model validation"""
_C.VALIDATION = CN()
_C.VALIDATION.BATCH_SIZE = 120


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
