#!/usr/bin/env python3

import pickle

import torch
from loguru import logger


pose_fpath = "debug/pose_validation_epoch000_iter000.pkl"
simcc_fpath = "debug/simcc_validation_epoch000_iter000.pkl"

with open(pose_fpath, "rb") as fin:
    pose = pickle.load(fin)
with open(simcc_fpath, "rb") as fin:
    simcc = pickle.load(fin)

pose_allpreds = pose['all_preds']
logger.info(f"len(pose.all_preds): {len(pose['all_preds'])}")
simcc_allpreds = simcc['all_preds']
logger.info(f"len(simcc.all_preds): {len(simcc['all_preds'])}")

assert len(pose['all_preds']) == len(simcc['all_preds'])

totally_allclosed = True
num_unmatched = 0
for i in range(len(pose['all_preds'])):
    _p = pose_allpreds[i]['pred']
    _s = torch.tensor(simcc_allpreds[i][:,:2], dtype=torch.float64)
    is_allclosed = torch.allclose(_p, _s)
    if is_allclosed is False:
        num_unmatched += 1
        logger.error(f"prediction result is not closed. idx({i}), image_id({pose_allpreds[i]['image_id']})")
        '''
        print("pose:")
        print(_p)
        print("simcc:")
        print(_s)
        print("diff")
        print(_p - _s)
        '''
    totally_allclosed = totally_allclosed and is_allclosed

logger.info(f"totally_allclosed: {totally_allclosed}")
logger.info(f"unmatched count: {num_unmatched:,} ({float(num_unmatched) / len(pose['all_preds']) * 100:.2f}%)")
logger.info("Done")
