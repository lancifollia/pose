#!/usr/bin/env python3

import pickle

import torch
from loguru import logger


pose_fpath = "compare/pose_batch_data.pkl"
simcc_fpath = "compare/simcc_batch_data.pkl"

with open(pose_fpath, "rb") as fin:
    pose = pickle.load(fin)

with open(simcc_fpath, "rb") as fin:
    simcc = pickle.load(fin)

assert len(pose['input']) == len(simcc['input']), "Unmatched batch size: pose({len(pose['input'])}) simcc({len(simcc['input'])})"
batch_size = len(pose['input'])
logger.info(f"Batch size: {batch_size}")

assert len(set(simcc['meta']['anno_id'].tolist()) - set(pose['meta']['anno_id'].tolist())) == 0
assert len(set(pose['meta']['anno_id'].tolist()) - set(simcc['meta']['anno_id'].tolist())) == 0

pose_dict = {}
for i in range(batch_size):
    v = {'input': pose['input'][i],
         'target_x': pose['target_x'][i],
         'target_y': pose['target_y'][i],
         'target_weight': pose['target_weight'][i],
         'anno_id': int(pose['meta']['anno_id'][i]),
         }
    pose_dict[int(pose['meta']['anno_id'][i])] = v

simcc_dict = {}
for i in range(batch_size):
    v = {'input': simcc['input'][i],
         'target_x': simcc['target_x'][i],
         'target_y': simcc['target_y'][i],
         'target_weight': simcc['target_weight'][i],
         'anno_id': int(simcc['meta']['anno_id'][i]),
         }
    simcc_dict[int(simcc['meta']['anno_id'][i])] = v

for anno_id in pose_dict.keys():
    logger.info(f"*** anno_id: {anno_id} ***")
    for attr in ['input', 'target_x', 'target_y', 'target_weight']:
        _pose = pose_dict[anno_id][attr]
        _simcc = simcc_dict[anno_id][attr]
        if torch.allclose(_pose.cpu(), _simcc.cpu()):
            logger.info(f"{attr} verification: pass")
        else:
            logger.error(f"{attr} verification: failed")

logger.info("Done")
