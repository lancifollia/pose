#!/usr/bin/env python3

import math
import pickle

import torch
from loguru import logger


def matching(a, b, keyword, verbose=False):
    exactly_matched = True
    allclosely_matched = True

    if (a - b).abs().sum() > 0.0:
        exactly_matched = False
        logger.error(f"[{keyword}] exact matching failed")
    else:
        if verbose:
            logger.info(f"[{keyword}] exact matching pass")

    if torch.allclose(a, b):
        if verbose:
            logger.info(f"[{keyword}] allclose matching pass")
    else:
        exactly_matched = False
        allclosely_matched = False
        logger.error(f"[{keyword}] allclose matching failed")

    return exactly_matched, allclosely_matched


def verify_batch_data(image_id, epoch, n_iter, verbose=False):
    if image_id is not None:
        pose_fpath = f"debug/{image_id}_pose_batch_data_epoch{epoch:03}_iter{n_iter:03}.pkl"
        simcc_fpath = f"debug/{image_id}_simcc_batch_data_epoch{epoch:03}_iter{n_iter:03}.pkl"
    else:
        pose_fpath = f"debug/pose_batch_data_epoch{epoch:03}_iter{n_iter:03}.pkl"
        simcc_fpath = f"debug/simcc_batch_data_epoch{epoch:03}_iter{n_iter:03}.pkl"

    with open(pose_fpath, "rb") as fin:
        pose = pickle.load(fin)
    with open(simcc_fpath, "rb") as fin:
        simcc = pickle.load(fin)

    if verbose:
        logger.info(f"pose  anno ids: {pose['meta']['anno_id']}")
        logger.info(f"simcc anno ids: {simcc['meta']['anno_id']}")
    is_annotation_id_same = True
    for p, s in zip(pose['meta']['anno_id'], simcc['meta']['anno_id']):
        if p != s:
            is_annotation_id_same = False
            logger.error(f"annotation id is not matched: pose({p}), simcc({s})")
    if len(pose['meta']['anno_id']) != len(simcc['meta']['anno_id']):
        logger.error(f"annotation id is not matched: pose({p}), simcc({s})")
        is_annotation_id_same = False
    if is_annotation_id_same:
        if verbose:
            logger.info("annotation id list is same.")
    else:
        logger.error("annotation id list is not same.")


    # Check
    exact_test_passed = True
    allclose_test_passed = True
    num_samples = pose['input'].shape[0]
    for i in range(num_samples):
        if verbose:
            logger.info(f"*** i={i}, anno_id: {pose['meta']['anno_id'][i]} ***")
        _exact_matched, _allclose_matched = matching(pose['meta']['bbox_center'][i].cpu(),
                                                     simcc['meta']['center'][i].cpu(),
                                                     f"meta.center,i={i}")
        exact_test_passed = exact_test_passed and _exact_matched
        allclose_test_passed = allclose_test_passed and _allclose_matched

        _exact_matched, _allclose_matched = matching(pose['meta']['bbox_scale'][i].cpu(),
                                                     simcc['meta']['scale'][i].cpu(),
                                                     f"meta.scale,i={i}")
        exact_test_passed = exact_test_passed and _exact_matched
        allclose_test_passed = allclose_test_passed and _allclose_matched

        _exact_matched, _allclose_matched = matching(pose['meta']['keypoints'][i].cpu(),
                                                     simcc['meta']['joints'][i,:,:2].cpu(),
                                                     f"meta.keypoints,i={i}")
        exact_test_passed = exact_test_passed and _exact_matched
        allclose_test_passed = allclose_test_passed and _allclose_matched

        for key in ['input', 'target_x', 'target_y', 'target_weight', 'output_x', 'output_y']:
            _exact_matched, _allclose_matched = matching(pose[key][i].cpu(), simcc[key][i].cpu(), f"{key},i={i}")
            exact_test_passed = exact_test_passed and _exact_matched
            allclose_test_passed = allclose_test_passed and _allclose_matched

    '''
    pf = pose['forward']
    sf = simcc['forward']
    for key in ['conv1', 'bn1', 'relu1', 'maxpool1',
                'layer1', 'layer2', 'layer3', 'layer4', 'final1', 'final2']:
        if (pf[key].cpu() - sf[key].cpu()).abs().sum() > 0.0:
            logger.error(f"[forward-{key}] exact matching failed")
        else:
            logger.info(f"[forward-{key}] exact matching pass")

        if torch.allclose(pf[key].cpu(), sf[key].cpu()):
            logger.info(f"[forward-{key}] allclose matching pass")
        else:
            logger.error(f"[forward-{key}] allclose matching failed")
            # print(pf[key] - sf[key])
    '''

    if math.fabs(pose['loss'] - simcc['loss']) > 0.0:
        exact_test_passed = False
        logger.error(f"[loss] exact matching failed - pose({pose['loss']}) simcc({simcc['loss']})")
    else:
        if verbose:
            logger.info(f"[loss] exact mathcing pass - pose({pose['loss']}) simcc({simcc['loss']})")

    if exact_test_passed is True:
        logger.info(f"epoch({epoch}) iter({n_iter}) Exact matching test passed")
    else:
        logger.error(f"epoch({epoch}) iter({n_iter}) Exact matching test failed")
        if allclose_test_passed is True:
            logger.info(f"epoch({epoch}) iter({n_iter}) Allclose matching test passed")
        else:
            logger.error(f"epoch({epoch}) iter({n_iter}) Allclose matching test failed")

    return exact_test_passed, allclose_test_passed


if __name__ == "__main__":
    """
    # for single image test
    image_ids = []
    # passed = [184613, 391895, 522418, 318219, 554625]
    # failed = []
    # slightly diff = []
    verify_batch_data('five', 0, 0, verbose=True)
    """

    exact_test_passed = True
    allclose_test_passed = True
    # for batch test
    for epoch in range(2):
        for it in range(31):
            _exact_passed, _allclose_passed = verify_batch_data(image_id=None, epoch=epoch, n_iter=it, verbose=False)
            exact_test_passed = exact_test_passed and _exact_passed
            allclose_test_passed = allclose_test_passed and _allclose_passed

    logger.info("--- Testing result ---")
    if exact_test_passed is True:
        logger.info("Exact matching test passed")
    else:
        logger.error("Exact matching test failed")
    if allclose_test_passed is True:
        logger.info("Allclose matching test passed")
    else:
        logger.error("Allclose matching test failed")
