#!/usr/bin/env python3

import pickle

import torch
from loguru import logger


pose_fpath = "model_compare/pose_init_model.pt"
simcc_fpath = "model_compare/simcc_init_model.pt"

with open(pose_fpath, "rb") as fin:
    pose = pickle.load(fin)

with open(simcc_fpath, "rb") as fin:
    simcc = pickle.load(fin)

assert len(pose) == len(simcc), "Unmatched parameter shape"

for i in range(len(pose)):
    _pose = pose[i]
    _simcc = simcc[i]

    if _pose.shape != _simcc.shape:
        logger.error(f"(i={i}) Unmatched shape: pose({_pose.shape}) simcc({_simcc.shape})")

    if torch.allclose(_pose.cpu(), _simcc.cpu()):
        pass
    else:
        logger.error(f"(i={i}) parameter is different")

logger.info("Verificatoin finished")
