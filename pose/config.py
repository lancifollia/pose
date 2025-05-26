from yacs.config import CfgNode as CN


_C = CN()

# CfgNodes should be alphabetical ordered.


"""Data Common"""
_C.DATA_COMMON = CN()
_C.DATA_COMMON.NUM_KEYPOINTS = 17  # COCO dataset
# Minimum required keypoints for training sample.


"""Data Loader"""
_C.DATA_LOADER = CN()
_C.DATA_LOADER.USE_PIN_MEMORY = True
_C.DATA_LOADER.BATCH_SIZE = 200
_C.DATA_LOADER.NUM_WORKERS = 24
_C.DATA_LOADER.TRAIN_DATA_SHUFFLE = True


"""Debugging"""
_C.DEBUG = CN()
_C.DEBUG.INITED_MODEL_PARAM = False
_C.DEBUG.TRAIN_DATA_SAMPLE = False
_C.DEBUG.VAL_DATA_SAMPLE = False
_C.DEBUG.FORWARD = False
_C.DEBUG.TRAINED_MODEL_PARAM = False


"""Training Data"""
_C.DATA_TRAIN = CN()
_C.DATA_TRAIN.ANNOTATION_FNAME = "/home/dev/data/coco/annotations/person_keypoints_train2017.json"
_C.DATA_TRAIN.IMAGE_DIR = "/home/dev/data/coco/images/train2017/"
# Restrict image_ids to use during training. (list of image_ids or null)
_C.DATA_TRAIN.TARGET_IMAGES = None


"""Training Data >> Data augmentation"""
_C.DATA_TRAIN.DATA_AUG = CN()
# Horizontal flipping trigering probability
_C.DATA_TRAIN.DATA_AUG.FLIP_PROBA = 0.5
# Image rotation trigering probability
_C.DATA_TRAIN.DATA_AUG.ROTATE_PROBA = 0.6
# Image rotating factor: angle in degrees (-180, 180)
_C.DATA_TRAIN.DATA_AUG.ROTATE_FACTOR = 40
# Image rescaling factor: ratio (0.0:do not scaling, 1.0:upto 200% scaling)
_C.DATA_TRAIN.DATA_AUG.SCALE_FACTOR = 0.3


"""Validation Data"""
_C.DATA_VAL = CN()
_C.DATA_VAL.ANNOTATION_FNAME = "/home/dev/data/coco/annotations/person_keypoints_val2017.json"
_C.DATA_VAL.IMAGE_DIR = "/home/dev/data/coco/images/val2017/"
_C.DATA_VAL.DATA_AUG = None
# Restrict image_ids to use during validation. (list of image_ids or null)
_C.DATA_VAL.TARGET_IMAGES = None


"""Model"""
_C.MODEL = CN()
_C.MODEL.NAME = "resnet"
# Neural network input image size. [width, height]
_C.MODEL.INPUT_SIZE_WH = [192, 256]
_C.MODEL.LOSS_TYPE = "KLDiscretLoss"

"""Model.Extra"""
_C.MODEL.EXTRA = CN()

"""ModelExtra > SimCC"""
_C.MODEL.EXTRA.SIMCC_SPLIT_FACTOR = 2.0
_C.MODEL.EXTRA.SIMCC_SIGMA = 1

"""Model.Extra > ResNet"""
# ResNet layer spec: 18, 34, 50, 101, 152
_C.MODEL.EXTRA.NUM_RESNET_LAYERS = 50
_C.MODEL.EXTRA.NUM_CHANNEL_PER_KP = 28
_C.MODEL.EXTRA.FINAL_CONV_KERNEL_SIZE = 1


"""Training"""
_C.TRAIN = CN()
_C.TRAIN.GPU_ID = 1
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 120]
_C.TRAIN.END_EPOCH = 140
_C.TRAIN.LOGGING_FREQ = 100
_C.TRAIN.SAVE_CHECKPOINTS = False
_C.TRAIN.LOGGING_EPOCH_PERF = False


"""Validation"""
_C.VALIDATE = CN()
_C.VALIDATE.LOGGING_FREQ = 100


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
