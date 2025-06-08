#!/usr/bin/env python3

import ujson as json


simcc_coco_fpath = "./debug/validation/simcc_coco_dump_epoch1"

with open(simcc_coco_fpath, "rb") as fin:
    _simcc_coco = json.load(fin)

result = []
for s in _simcc_coco:
    v = {"image_id": s["image_id"],
         "category_id": 1,
         "score": 1.0,
         "keypoints": s["keypoints"]
         }
    result.append(v)

result_fpath = "./debug/validation/simcc_coco_dump_epoch1.reformat"
with open(result_fpath, 'w') as fout:
    fout.write(json.dumps(result))

print("Done")
