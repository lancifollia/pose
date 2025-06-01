#!/usr/bin/env python3

import math
import pickle

import torch
from loguru import logger


def matching(a, b, keyword, verbose=False):
    exactly_matched = True
    allclosely_matched = True

    if (a - b).abs().sum() > 0.0:
        exactly_matched = False
        logger.error(f"[{keyword}] exact matching failed")
    else:
        if verbose:
            logger.info(f"[{keyword}] exact matching pass")

    if torch.allclose(a, b):
        if verbose:
            logger.info(f"[{keyword}] allclose matching pass")
    else:
        exactly_matched = False
        allclosely_matched = False
        logger.error(f"[{keyword}] allclose matching failed")

    return exactly_matched, allclosely_matched


# Configurations
pose_model_path = "/home/dev/pose/checkpoints/pose_fulliter_checkpoint_1.pth"
simcc_model_path = "/home/dev/simcc/output/20250601_fulliter_2epoch_no_dataaug/final_state.pth"


# 1. Verify model parameter
pose_model = torch.load(pose_model_path)['model_state_dict']
simcc_model = torch.load(simcc_model_path)

exact_match_pass = True
allclose_match_pass = True

for layer in pose_model.keys():
    if 'cc_head' in layer:
        simcc_layer_name = layer.replace("cc", "mlp")
        _exact_pass, _allclose_pass = matching(pose_model[layer].cpu(),
                                               simcc_model[simcc_layer_name].cpu(),
                                               layer, verbose=False)
    else:
        _exact_pass, _allclose_pass = matching(pose_model[layer].cpu(),
                                               simcc_model[layer].cpu(),
                                               layer, verbose=False)

    exact_match_pass = exact_match_pass and _exact_pass
    allclose_match_pass = allclose_match_pass and _allclose_pass

logger.info("--- Model paramer matching result ---")
if exact_match_pass is True:
    logger.info("Exact matching test passed")
else:
    logger.error("Exact matching test failed")
if allclose_match_pass is True:
    logger.info("Allclose matching test passed")
else:
    logger.error("Allclose matching test failed")


# 2. Verify validation loss
pose_fpath = "/home/dev/pose/debug/pose_validation_epoch000_iter000.pkl"
simcc_fpath = "/home/dev/pose/debug/simcc_validation_epoch000_iter000.pkl"

with open(pose_fpath, "rb") as fin:
    pose = pickle.load(fin)
with open(simcc_fpath, "rb") as fin:
    simcc = pickle.load(fin)

pose_val_loss = pose['val_loss_mean']
simcc_val_loss = simcc['val_loss_mean']

if math.fabs(pose_val_loss - simcc_val_loss) > 0.0:
    logger.error("validation loss matching failed")
    logger.error(f"pose({pose_val_loss}), simcc({simcc_val_loss})")
else:
    logger.info("validation loss matching passed")


# 3. Verify APs, ARs
# 4. Verify prediction result



logger.info("Done")
