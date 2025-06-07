#!/usr/bin/env python3

import pickle

import numpy as np
import torch
import ujson as json
from loguru import logger


# parameters
verbose = False
skip_image_ids = set([43314, 459272])
logger.info(f"skip image ids(size:{len(skip_image_ids)}): {skip_image_ids}")


"""
Load allpred data
"""
pose_ap_fpath = "debug/pose_validation_epoch000_iter000.pkl"
simcc_ap_fpath = "debug/simcc_validation_epoch000_iter000.pkl"

with open(pose_ap_fpath, "rb") as fin:
    _pose_ap = pickle.load(fin)
with open(simcc_ap_fpath, "rb") as fin:
    _simcc_ap = pickle.load(fin)

logger.info(f"(allpreds) original data size: pose({len(_pose_ap['all_preds']):,}) simcc({len(_simcc_ap['all_preds']):,})")
pose_ap = [x for x in _pose_ap['all_preds'] if x['image_id'] not in skip_image_ids]
simcc_ap = []
for i in range(len(_simcc_ap['image_path'])):
    # image_path example: "data/dataset/coco/images/val2017/000000168974.jpg"
    image_id = int(_simcc_ap['image_path'][i].split('/')[-1].split('.')[0])
    if image_id in skip_image_ids:
        continue
    simcc_ap.append(torch.tensor(_simcc_ap['all_preds'][i][:,:2], dtype=torch.float64))
logger.info(f"len(pose.all_preds): {len(pose_ap)}")
logger.info(f"len(simcc.all_preds): {len(simcc_ap)}")
assert len(pose_ap) == len(simcc_ap)
logger.info(f"(allpreds) data size(after removing skip images): pose({len(pose_ap):,}) simcc({len(simcc_ap):,})")


"""
Load coco dump data
"""
pose_coco_fpath = "./debug/pred_dump/pose_pred_dump_epoch1"
simcc_coco_fpath = "./debug/pred_dump/simcc_pred_dump_epoch1"

with open(pose_coco_fpath, "rb") as fin:
    _pose_coco = json.load(fin)
with open(simcc_coco_fpath, "rb") as fin:
    _simcc_coco = json.load(fin)

logger.info(f"(COCO val dump) original data size: pose({len(_pose_coco):,}) simcc({len(_simcc_coco):,})")
pose_coco = [x for x in _pose_coco if x['image_id'] not in skip_image_ids]
simcc_coco = [x for x in _simcc_coco if x['image_id'] not in skip_image_ids]
logger.info(f"(COCO val dump) data size(after removing skip images): pose({len(pose_coco):,}) simcc({len(simcc_coco):,})")
assert len(pose_coco) == len(simcc_coco)


"""
Compare pose.allpred vs. simcc.allpred
"""
logger.info("=== Compare pose.allpred vs. simcc.allpred ===")
num_unmatched = 0
total_count = len(pose_ap)
for i in range(total_count):
    _p = pose_ap[i]['pred']
    _s = simcc_ap[i]
    if torch.allclose(_p, _s) is False:
        num_unmatched += 1
        if verbose:
            logger.error(f"prediction result is not closed. idx({i}), image_id({pose_ap[i]['image_id']})")
            print("pose:")
            print(_p)
            print("simcc:")
            print(_s)
            print("diff")
            print(_p - _s)
unmatching_perc = float(num_unmatched) / total_count * 100
logger.info(f"unmatched/total = {num_unmatched:,}/{total_count:,} ({unmatching_perc:.2f}%)")


"""
Compare pose.allpred vs. pose.coco_dump
"""
logger.info("=== Compare pose.allpred vs. pose.coco_dump ===")
num_unmatched = 0
total_count = len(pose_ap)
for i in range(total_count):
    pred = pose_ap[i]['pred']
    kps = torch.tensor(np.asarray(pose_coco[i]['keypoints']).reshape(17,3)[:,:2])
    if torch.allclose(pred, kps) is False:
        num_unmatched += 1
unmatching_perc = float(num_unmatched) / total_count * 100
logger.info(f"unmatched/total = {num_unmatched:,}/{total_count:,} ({unmatching_perc:.2f}%)")


"""
Compare simcc.allpred vs. simcc.coco_dump
"""
logger.info("=== Compare simcc.allpred vs. simcc.coco_dump ===")
num_unmatched = 0
total_count = len(simcc_ap)
for i in range(total_count):
    pred = simcc_ap[i]
    kps = torch.tensor(np.asarray(simcc_coco[i]['keypoints']).reshape(17,3)[:,:2])
    if torch.allclose(pred, kps) is False:
        num_unmatched += 1
unmatching_perc = float(num_unmatched) / total_count * 100
logger.info(f"unmatched/total = {num_unmatched:,}/{total_count:,} ({unmatching_perc:.2f}%)")


"""
Compare pose.coco_dump vs. simcc.coco_dump
"""
logger.info("=== Compare pose.coco_dump vs. simcc.coco_dump ===")
num_unmatched = 0
total_count = len(pose_coco)
for i in range(total_count):
    assert pose_coco[i]['image_id'] == simcc_coco[i]['image_id']
    pkps = np.asarray(pose_coco[i]['keypoints']).reshape(17,3)[:,:2]
    skps = np.asarray(simcc_coco[i]['keypoints']).reshape(17,3)[:,:2]
    if not np.allclose(pkps, skps):
        num_unmatched += 1
        if verbose:
            print("pose:")
            print(pkps)
            print("simcc:")
            print(skps)
            print("diff:")
            print(pkps - skps)

unmatching_perc = float(num_unmatched) / total_count * 100
logger.info(f"unmatched/total = {num_unmatched:,}/{total_count:,} ({unmatching_perc:.2f}%)")
logger.info("Done")
