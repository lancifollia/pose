#!/usr/bin/env python3


from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


anno_fname = "/home/dev/data/coco/annotations/person_keypoints_val2017.json"
stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)',
               'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']


def coco_eval(pred_dump_fpath):
    coco_gt = COCO(anno_fname)
    coco_dt = coco_gt.loadRes(pred_dump_fpath)
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    # coco_eval.params.imgIds = list(eval_target_image_ids)
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # for i, s in enumerate(stats_names):
    #     print(f"{s}: {coco_eval.stats[i]:.4f}")


pred_fname = "debug/validation/simcc_coco_dump_epoch1"
logger.info(f"pred file: {pred_fname}")
coco_eval(pred_fname)

pred_fname = "debug/validation/simcc_coco_dump_epoch1.reformat"
logger.info(f"pred file: {pred_fname}")
coco_eval(pred_fname)

pred_fname = "debug/validation/pose_coco_dump_epoch1"
logger.info(f"pred file: {pred_fname}")
coco_eval(pred_fname)
