#!/usr/bin/env python3

import pickle

import torch
from loguru import logger


def verify_parameters(pose_fpath, simcc_fpath, category):
    logger.info(f"=-= Start model parameter verification: [{category}] =-=")
    verified = True

    with open(pose_fpath, "rb") as fin:
        pose = pickle.load(fin)

    with open(simcc_fpath, "rb") as fin:
        simcc = pickle.load(fin)

    assert len(pose) == len(simcc), "Unmatched parameter shape"

    for i in range(len(pose)-1, -1, -1):
        _pose = pose[i]
        _simcc = simcc[i]

        # print("i:", i)
        # import pdb
        # pdb.set_trace()

        if _pose.shape != _simcc.shape:
            logger.error(f"[{category}](i={i}) Unmatched shape: pose({_pose.shape}) simcc({_simcc.shape})")

        # if torch.allclose(_pose.cpu(), _simcc.cpu(), atol=1e-05, rtol=1e-04):
        if torch.allclose(_pose.cpu(), _simcc.cpu(), atol=1e-10, rtol=1e-08):
            logger.info(f"[{category}](i={i}) parameter is closed")
        else:
            logger.error(f"[{category}](i={i}) parameter is different")
            verified = False

    if verified:
        logger.info(f"[{category}] verification finished")
    else:
        logger.error(f"[{category}] verification failed")


# """
verify_parameters(pose_fpath="compare/pose_init_model.pt",
                  simcc_fpath="compare/simcc_init_model.pt",
                  category="init model"
                  )
# """

verify_parameters(pose_fpath="compare/pose_1st_epoch_model.pt",
                  simcc_fpath="compare/simcc_1st_epoch_model.pt",
                  category="1st_epoch"
                  )
