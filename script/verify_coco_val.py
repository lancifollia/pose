#!/usr/bin/env python3

import pickle
from collections import defaultdict

import numpy as np
import torch
import ujson as json
from loguru import logger


# parameters
verbose = False
# skip_image_ids = set([])
skip_image_ids = set([43314, 459272])
logger.info(f"skip image ids(size:{len(skip_image_ids)}): {skip_image_ids}")


"""
Load allpred data
"""
pose_ap_fpath = "debug/validation/pose_validation_epoch001_iter000.pkl"
simcc_ap_fpath = "debug/validation/simcc_validation_epoch001_iter000.pkl"
# simcc_ap_fpath = "debug/validation/simcc_validation_epoch001_iter000.pkl.no_nms"

with open(pose_ap_fpath, "rb") as fin:
    _pose_ap_raw = pickle.load(fin)
with open(simcc_ap_fpath, "rb") as fin:
    _simcc_ap_raw = pickle.load(fin)

logger.info(f"(allpreds) original data size: pose({len(_pose_ap_raw['all_preds']):,}) simcc({len(_simcc_ap_raw['all_preds']):,})")
_pose_ap = defaultdict(list)
num_processed_pose = 0
for i in range(len(_pose_ap_raw['all_preds'])):
    image_id = _pose_ap_raw['all_preds'][i]['image_id']
    if image_id in skip_image_ids:
        continue
    _pose_ap[image_id].append(_pose_ap_raw['all_preds'][i]['pred'])
    num_processed_pose += 1
pose_ap = {}
for image_id, preds in _pose_ap.items():
    pose_ap[image_id] = sorted(preds, key=lambda x: x[0][0])

_simcc_ap = defaultdict(list)
num_processed_simcc = 0
for i in range(len(_simcc_ap_raw['image_path'])):
    # image_path example: "data/dataset/coco/images/val2017/000000168974.jpg"
    image_id = int(_simcc_ap_raw['image_path'][i].split('/')[-1].split('.')[0])
    if image_id in skip_image_ids:
        continue
    _simcc_ap[image_id].append(_simcc_ap_raw['all_preds'][i][:,:2])
    num_processed_simcc += 1
simcc_ap = {}
for image_id, preds in _simcc_ap.items():
    simcc_ap[image_id] = sorted(preds, key=lambda x: x[0][0])

logger.info(f"len(pose.all_preds): {len(pose_ap)}")
logger.info(f"len(simcc.all_preds): {len(simcc_ap)}")
logger.info(f"num processed pose, simcc: {num_processed_pose:,}, {num_processed_simcc:,}")
assert len(pose_ap) == len(simcc_ap)
assert num_processed_pose == num_processed_simcc


"""
Load coco dump data
"""
pose_coco_fpath = "./debug/validation/pose_coco_dump_epoch1"
simcc_coco_fpath = "./debug/validation/simcc_coco_dump_epoch1"
# simcc_coco_fpath = "./debug/validation/simcc_coco_dump_epoch1.no_nms"

with open(pose_coco_fpath, "rb") as fin:
    _pose_coco_raw = json.load(fin)
with open(simcc_coco_fpath, "rb") as fin:
    _simcc_coco_raw = json.load(fin)

logger.info(f"(COCO val dump) original data size: pose({len(_pose_coco_raw):,}) simcc({len(_simcc_coco_raw):,})")
_pose_coco = defaultdict(list)
num_processed_pose = 0
for i in range(len(_pose_coco_raw)):
    image_id = _pose_coco_raw[i]['image_id']
    if image_id in skip_image_ids:
        continue
    _pose_coco[image_id].append(np.asarray(_pose_coco_raw[i]['keypoints']).reshape(17,3)[:,:2])
    num_processed_pose += 1
pose_coco = {}
for image_id, kpts in _pose_coco.items():
    pose_coco[image_id] = sorted(kpts, key=lambda x: x[0][0])

_simcc_coco = defaultdict(list)
num_processed_simcc = 0
for i in range(len(_simcc_coco_raw)):
    image_id = _simcc_coco_raw[i]['image_id']
    if image_id in skip_image_ids:
        continue
    _simcc_coco[image_id].append(np.asarray(_simcc_coco_raw[i]['keypoints']).reshape(17,3)[:,:2])
    num_processed_simcc += 1
simcc_coco = {}
for image_id, kpts in _simcc_coco.items():
    simcc_coco[image_id] = sorted(kpts, key=lambda x: x[0][0])

logger.info(f"(COCO val dump) data size(after removing skip images): pose({len(pose_coco):,}) simcc({len(simcc_coco):,})")
logger.info(f"num processed pose, simcc: {num_processed_pose:,}, {num_processed_simcc:,}")
assert len(pose_coco) == len(simcc_coco)
assert num_processed_pose == num_processed_simcc


"""
Compare pose.allpred vs. simcc.allpred
"""
logger.info("=== Compare pose.allpred vs. simcc.allpred ===")
num_unmatched = 0
num_try_matching = 0
image_ids = pose_ap.keys()
for image_id in image_ids:
    _ps = pose_ap[image_id]
    _ss = simcc_ap[image_id]
    assert len(_ps) == len(_ss)
    for i in range(len(_ps)):
        _p = _ps[i]
        _s = _ss[i]
        if np.allclose(_p, _s) is False:
            num_unmatched += 1
            if verbose:
                logger.error(f"prediction result is not closed. image({image_id}), i({i})")
                print("pose:")
                print(_p)
                print("simcc:")
                print(_s)
                print("diff")
                print(_p - _s)
        num_try_matching += 1
unmatching_perc = float(num_unmatched) / num_try_matching * 100
logger.info(f"unmatched/try_matching = {num_unmatched:,}/{num_try_matching:,} ({unmatching_perc:.2f}%)")


"""
Compare pose.allpred vs. pose.coco_dump
"""
logger.info("=== Compare pose.allpred vs. pose.coco_dump ===")
num_unmatched = 0
num_try_matching = 0
image_ids = pose_ap.keys()
for image_id in image_ids:
    _preds = pose_ap[image_id]
    _kpss = pose_coco[image_id]
    assert len(_preds) == len(_kpss)
    for i in range(len(_preds)):
        _pred = _preds[i]
        _kps = _kpss[i]
        if np.allclose(_pred, _kps) is False:
            num_unmatched += 1
        num_try_matching += 1
unmatching_perc = float(num_unmatched) / num_try_matching * 100
logger.info(f"unmatched/try_matching = {num_unmatched:,}/{num_try_matching:,} ({unmatching_perc:.2f}%)")


"""
Compare simcc.allpred vs. simcc.coco_dump
"""
logger.info("=== Compare simcc.allpred vs. simcc.coco_dump ===")
num_unmatched = 0
num_try_matching = 0
image_ids = simcc_ap.keys()
for image_id in image_ids:
    _preds = simcc_ap[image_id]
    _kpss = simcc_coco[image_id]
    assert len(_preds) == len(_kpss)
    for i in range(len(_preds)):
        _pred = _preds[i]
        _kps = _kpss[i]
        if np.allclose(_pred, _kps) is False:
            num_unmatched += 1
        num_try_matching += 1
unmatching_perc = float(num_unmatched) / num_try_matching * 100
logger.info(f"unmatched/try_matching = {num_unmatched:,}/{num_try_matching:,} ({unmatching_perc:.2f}%)")


"""
Compare pose.coco_dump vs. simcc.coco_dump
"""
logger.info("=== Compare pose.coco_dump vs. simcc.coco_dump ===")
num_unmatched = 0
num_try_matching = 0
assert len(pose_coco) == len(simcc_coco)
image_ids = pose_coco.keys()
for image_id in image_ids:
    _ps = pose_coco[image_id]
    _ss = simcc_coco[image_id]
    assert len(_ps) == len(_ss)
    for i in range(len(_ps)):
        _p = _ps[i]
        _s = _ss[i]
        if np.allclose(_p, _s) is False:
            num_unmatched += 1
        num_try_matching += 1
unmatching_perc = float(num_unmatched) / num_try_matching * 100
logger.info(f"unmatched/try_matching = {num_unmatched:,}/{num_try_matching:,} ({unmatching_perc:.2f}%)")

logger.info("Done")
