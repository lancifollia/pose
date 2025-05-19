#!/usr/bin/env python3

import math
import pickle

import torch
from loguru import logger


def matching(a, b, keyword):
    if (a - b).abs().sum() > 0.0:
        logger.warning(f"[{keyword}] exact matching failed")
        # print(a - b)
    else:
        logger.info(f"[{keyword}] exact matching pass")

    if torch.allclose(a, b):
        logger.info(f"[{keyword}] allclose matching pass")
    else:
        logger.warning(f"[{keyword}] allclose matching failed")


def verify_batch_data(image_id, epoch, n_iter):
    # pose_fpath = f"debug/{image_id}_pose_batch_data_epoch{epoch:03}_iter{n_iter:03}.pkl"
    pose_fpath = f"debug/pose_batch_data_epoch{epoch:03}_iter{n_iter:03}.pkl"
    simcc_fpath = f"debug/{image_id}_simcc_batch_data_epoch{epoch:03}_iter{n_iter:03}.pkl"

    with open(pose_fpath, "rb") as fin:
        pose = pickle.load(fin)
    with open(simcc_fpath, "rb") as fin:
        simcc = pickle.load(fin)


    logger.info(f"pose  anno ids: {pose['meta']['anno_id']}")
    logger.info(f"simcc anno ids: {simcc['meta']['anno_id']}")
    is_annotation_id_same = True
    for p, s in zip(pose['meta']['anno_id'], simcc['meta']['anno_id']):
        if p != s:
            is_annotation_id_same = False
            logger.warning(f"annotation id is not matched: pose({p}), simcc({s})")
    if len(pose['meta']['anno_id']) != len(simcc['meta']['anno_id']):
        logger.warning(f"annotation id is not matched: pose({p}), simcc({s})")
        is_annotation_id_same = False
    if is_annotation_id_same:
        logger.info("annotation id list is same.")
    else:
        logger.warning("annotation id list is not same.")


    # Check
    num_samples = pose['input'].shape[0]
    for i in range(num_samples):
        logger.info(f"*** i={i} ***")
        matching(pose['meta']['bbox_center'][i].cpu(),
                 simcc['meta']['center'][i].cpu(),
                 f"meta.center,i={i}")
        matching(pose['meta']['bbox_scale'][i].cpu() / 200,  # 200: pixel std of SimCC code
                 simcc['meta']['scale'][i].cpu(),
                 f"meta.scale,i={i}")
        matching(pose['meta']['keypoints'][i].cpu(),
                 simcc['meta']['joints'][i,:,:2].cpu(),
                 f"meta.keypoints,i={i}")
        for key in ['image_np', 'trans', 'input_phase1']:
            matching(pose['meta'][key][i].cpu(), simcc['meta'][key][i].cpu(), f"meta.{key},i={i}")
        for key in ['input', 'target_x', 'target_y', 'target_weight', 'output_x', 'output_y']:
            matching(pose[key][i].cpu(), simcc[key][i].cpu(), f"{key},i={i}")
        logger.info("Skip other batch data")


    pf = pose['forward']
    sf = simcc['forward']
    for key in ['conv1', 'bn1', 'relu1', 'maxpool1',
                'layer1', 'layer2', 'layer3', 'layer4', 'final1', 'final2']:
        if (pf[key].cpu() - sf[key].cpu()).abs().sum() > 0.0:
            logger.warning(f"[forward-{key}] exact matching failed")
        else:
            logger.info(f"[forward-{key}] exact matching pass")

        if torch.allclose(pf[key].cpu(), sf[key].cpu()):
            logger.info(f"[forward-{key}] allclose matching pass")
        else:
            logger.warning(f"[forward-{key}] allclose matching failed")
            # print(pf[key] - sf[key])


    if math.fabs(pose['loss'] - simcc['loss']) > 0.0:
        logger.warning(f"[loss] exact matching failed - pose({pose['loss']}) simcc({simcc['loss']})")
    else:
        logger.info(f"[loss] exact mathcing pass - pose({pose['loss']}) simcc({simcc['loss']})")

    print("trans(pose):", pose['meta']['trans'][0])
    print("trans(simc):", simcc['meta']['trans'][0])

    import pdb
    pdb.set_trace()
    print("debug")


if __name__ == "__main__":
    image_ids = [184613]
    # passed = [391895, 522418]
    # failed = [184613]
    # slightly diff = [318219, 554625]
    verify_batch_data(image_ids[0], 0, 0)
