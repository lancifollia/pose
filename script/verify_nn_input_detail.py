#!/usr/bin/env python3

import pickle
import math

import numpy as np
from loguru import logger


ABS_TOLERANCE = 1e-10


def compare(a, b, subject):
    if math.fabs((a-b).sum()) > 0.0:
        logger.error(f"[exact - failed] {subject}")
    else:
        logger.info(f"[exact - pass] {subject}")

    if np.allclose(a, b, atol=ABS_TOLERANCE):
        logger.info(f"[close - pass] {subject}")
    else:
        logger.error(f"[close - failed] {subject}")
        print(f"isclose:\n{np.isclose(a, b)}")
        print(f"\npose:\n{a}")
        print(f"\nsimcc:\n{b}")


pose_fpath = "compare/pose_nn_input_detail.pkl"
simcc_fpath = "compare/simcc_nn_input_detail.pkl"

with open(pose_fpath, "rb") as fin:
    pose = pickle.load(fin)

with open(simcc_fpath, "rb") as fin:
    simcc = pickle.load(fin)

pose_image_np = pose['image_numpy']
simcc_image_np = simcc['image_numpy']

# Image numpy
if np.allclose(pose_image_np, simcc_image_np, atol=ABS_TOLERANCE):
    logger.info("[pass] image numpy verification")
else:
    logger.error("[failed] image numpy verification")

# Origin keypoints
pose_kps = pose['origin_kps']
simcc_kps = simcc['origin_kps']
if np.allclose(pose_kps, simcc_kps[:,:2], atol=ABS_TOLERANCE):
    logger.info("[pass] original joint data verification")
else:
    logger.error("[failed] original joint data verification")

# Origin visibility
pose_vis = pose['origin_vis']
pose['origin_vis'][pose['origin_vis'] == 2] = 1.0
simcc_vis = simcc['origin_vis'][:,0]
if np.allclose(pose_vis, simcc_vis, atol=ABS_TOLERANCE):
    logger.info("[pass] original vis data verification")
else:
    logger.error("[failed] original vis data verification")
# Affine transform matrix
pose_trans = pose['trans']
simcc_trans = simcc['trans']
if np.allclose(pose_trans, simcc_trans, atol=ABS_TOLERANCE):
    logger.info("[pass] trans matrix verification")
else:
    logger.error("[failed] trans matrix verification")
    print(f"isclose:\n{np.isclose(pose_trans, simcc_trans)}")
    print(f"\npose_trans:\n{pose_trans}")
    print(f"\nsimcc_trans:\n{simcc_trans}")

"""
# (detail) affine transform matrix
pose_affine_fpath = "compare/pose_affine.pkl"
simcc_affine_fpath = "compare/simcc_affine.pkl"
with open(pose_affine_fpath, "rb") as fin:
    pose_affine = pickle.load(fin)
with open(simcc_affine_fpath, "rb") as fin:
    simcc_affine = pickle.load(fin)

compare(pose_affine['src'], simcc_affine['src'], "affine src")
compare(pose_affine['dst'], simcc_affine['dst'], "affine dst")
compare(pose_affine['trans'], simcc_affine['trans'], "affine trans")
"""

compare(pose['origin_input'], simcc['origin_input'], 'origin input')
compare(pose['final_kps'], simcc['final_kps'][:,:2], 'final kps')
compare(pose['final_input'], simcc['final_input'], 'final input')

#import pdb
#pdb.set_trace()

compare(pose['target_x'], simcc['target_x'], 'target x')
compare(pose['target_y'], simcc['target_y'], 'target y')
compare(pose['target_weight'], simcc['target_weight'], 'target weight')

# import pdb
# pdb.set_trace()

print("Done")
