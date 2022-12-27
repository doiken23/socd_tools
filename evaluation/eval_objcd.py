#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from pycocoobjcdtools.coco import COCO
from pycocoobjcdtools.cocoeval import COCOObjcdEval

from coco_eval import evaluate_objcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_path", type=str)
    parser.add_argument("dt_path", type=str)
    parser.add_argument("--use_mask", action="store_true", default=False)
    args = parser.parse_args()

    coco_gt = COCO(args.gt_path)
    coco_dt = coco_gt.loadRes(args.dt_path)

    if args.use_mask:
        iou_types = ["bbox", "segm"]
    else:
        iou_types = ["bbox"]

    for iou_type in iou_types:
        coco_eval = COCOObjcdEval(cocoGt=coco_gt, cocoDt=coco_dt, iouType=iou_type)
        coco_eval.params.areaRng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
            [32**2, 1e5**2],
        ]
        coco_eval.params.areaRngLbl = ["all", "small", "medium", "large", "not small"]

        coco_eval.evaluate()
        coco_eval.accumulate()

        results = evaluate_objcd(coco_eval)
        out_results = {}
        for k, v in results.items():
            if k.startswith("precision") or k.startswith("recall"):
                out_results[k] = v
            if k.startswith("precision"):
                precision = v
                recall = results[k.replace("precision", "recall")]
                f1_score = 2 * recall * precision / (recall + precision)
                out_results[k.replace("precision", "f1")] = f1_score

        if "carla_cd" in args.gt_path:
            from datasets.carla_dataset import LABEL_NAMES
        elif "gsv_cd" in args.gt_path:
            from datasets.gsv_dataset import LABEL_NAMES

            LABEL_NAMES = ["Construction", "Object", "Vehicle"] + LABEL_NAMES

        df = pd.DataFrame(out_results, index=LABEL_NAMES).T

        header = "ignore_" if args.gt_path.split("/")[-1].startswith("ignore_") else ""
        if iou_type == "bbox":
            file_name = "{}results.csv".format(header)
        else:
            file_name = "{}mask_results.csv".format(header)
        df.to_csv(Path(args.dt_path).parent.joinpath(file_name))


if __name__ == "__main__":
    main()
