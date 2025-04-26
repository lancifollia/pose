#!/usr/bin/env python3

import math
import pickle

import torch
from loguru import logger


def verify_batch_data(epoch, n_iter):
    pose_fpath = f"debug/pose_batch_data_epoch{epoch:03}_iter{n_iter:03}.pkl"
    simcc_fpath = f"debug/simcc_batch_data_epoch{epoch:03}_iter{n_iter:03}.pkl"

    with open(pose_fpath, "rb") as fin:
        pose = pickle.load(fin)
    with open(simcc_fpath, "rb") as fin:
        simcc = pickle.load(fin)


    logger.info(f"pose  anno ids: {pose['meta']['anno_id']}")
    logger.info(f"simcc anno ids: {simcc['meta']['anno_id']}")
    is_annotation_id_same = True
    for p, s in zip(pose['meta']['anno_id'], simcc['meta']['anno_id']):
        if p != s:
            is_annotation_id_same = False
            logger.warning(f"annotation id is not matched: pose({p}), simcc({s})")
    if len(pose['meta']['anno_id']) != len(simcc['meta']['anno_id']):
        logger.warning(f"annotation id is not matched: pose({p}), simcc({s})")
        is_annotation_id_same = False
    if is_annotation_id_same:
        logger.info("annotation id list is same.")
    else:
        logger.warning("annotation id list is not same.")


    # center
    simcc_center = simcc['meta']['center']
    pose_center = pose['meta']['bbox_center']
    if (pose_center - simcc_center).abs().sum() > 0.0:
        logger.warning("[bbox center] exact matching failed")
        logger.warning(f"pose bbox center:\n{pose_center}")
        logger.warning(f"simcc bbox center:\n{simcc_center}")
    else:
        logger.info("[bbox center] exact matching pass")

    if torch.allclose(pose_center, simcc_center):
        logger.info("[bbox center] allclose matching pass")
    else:
        logger.warning("[bbox center] allclose matching failed")


    # TODO: scale
    # TODO: keypoint


    # Check input feature, target, output
    for key in ['input', 'target_x', 'target_y', 'target_weight', 'output_x', 'output_y']:
        if (pose[key].cpu() - simcc[key].cpu()).abs().sum() > 0.0:
            logger.warning(f"[{key}] exact matching failed")
        else:
            logger.info(f"[{key}] exact matching pass")

        if torch.allclose(pose[key].cpu(), simcc[key].cpu()):
            logger.info(f"[{key}] allclose matching pass")
        else:
            logger.warning(f"[{key}] allclose matching failed")

    if math.fabs(pose['loss'] - simcc['loss']) > 0.0:
        logger.warning(f"[loss] exact matching failed - pose({pose['loss']}) simcc({simcc['loss']})")
    else:
        logger.info(f"[loss] exact mathcing pass - pose({pose['loss']}) simcc({simcc['loss']})")


    import pdb
    pdb.set_trace()
    print("debug")


if __name__ == "__main__":
    verify_batch_data(0, 0)
