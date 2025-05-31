#!/usr/bin/env python3

import math
import pickle

import numpy
from loguru import logger


def matching_scalar(a, b, keyword):
    if math.fabs(a - b) > 0.0:
        logger.warning(f"(scalar) [{keyword}] exact matching failed")
        logger.warning(f"a: {a}, b: {b}")
        logger.warning(f"diff(a-b): {a-b}")
    else:
        logger.info(f"(scalar) [{keyword}] exact matching pass")


def matching_numpy(a, b, keyword):
    if numpy.fabs(a - b).sum() > 0.0:
        logger.warning(f"[{keyword}] exact matching failed")
        # print(a - b)
    else:
        logger.info(f"[{keyword}] exact matching pass")

    if numpy.allclose(a, b):
        logger.info(f"[{keyword}] allclose matching pass")
    else:
        logger.warning(f"[{keyword}] allclose matching failed")


pose_fpath = "debug/pose_trans.pkl"
simcc_fpath = "debug/simcc_trans.pkl"

with open(pose_fpath, "rb") as fin:
    pose = pickle.load(fin)
with open(simcc_fpath, "rb") as fin:
    simcc = pickle.load(fin)

# scalar
for key in ['rot', 'src_w', 'dst_w', 'dst_h', 'rot_rad']:
    matching_scalar(pose[key], simcc[key], key)

# numpy
for key in ['center', 'scale', 'output_size', 'src_dir', 'dst_dir', 'src', 'dst', 'trans']:
    matching_numpy(pose[key], simcc[key], key)

logger.info("Done")
