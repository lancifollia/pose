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

    for key in ['input', 'target_x', 'target_y', 'target_weight', 'output_x', 'output_y']:
        if (pose[key].cpu() - simcc[key].cpu()).abs().sum() > 0.0:
            logger.warning(f"[exact failed({key})]")
        else:
            logger.info(f"[exact pass({key})]")

        if torch.allclose(pose[key].cpu(), simcc[key].cpu()):
            logger.info(f"[allclose pass({key})]")
        else:
            logger.warning(f"[allclose failed({key})]")

    if math.fabs(pose['loss'] - simcc['loss']) > 0.0:
        logger.warning(f"[exact failed(loss)] pose({pose['loss']}) simcc({simcc['loss']})")
    else:
        logger.info(f"[exact pass(loss)] pose({pose['loss']}) simcc({simcc['loss']})")

    # TODO: compare image, annotation ids

    #import pdb
    #pdb.set_trace()
    #print("AA")


if __name__ == "__main__":
    verify_batch_data(0, 0)
