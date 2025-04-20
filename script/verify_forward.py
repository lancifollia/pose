#!/usr/bin/env python3

import pickle

import torch
from loguru import logger


def verify_forward(pose_fpath, simcc_fpath, category):
    logger.info(f"=-= Start forward verification: [{category}] =-=")
    verified = True

    with open(pose_fpath, "rb") as fin:
        pose = pickle.load(fin)

    with open(simcc_fpath, "rb") as fin:
        simcc = pickle.load(fin)

    for data in ["output_x", "output_y"]:
        if torch.allclose(pose[data], simcc[data]):
            pass
        else:
            logger.error(f"[{category},{data}] is different")
            verified = False

    if verified:
        logger.info(f"[{category}] verification finished")
    else:
        logger.error(f"[{category}] verification failed")


verify_forward(pose_fpath="compare/pose_forward.pkl",
               simcc_fpath="compare/simcc_forward.pkl",
               category="forward"
               )
