#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from pprint import pprint

import numpy as np
from chainercv.evaluations import eval_semantic_segmentation
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_dir", type=str, help="GT directory path")
    parser.add_argument("pred_dir", type=str, help="prediction result directory path")
    parser.add_argument("output_dir", type=str, help="output directory path")
    parser.add_argument(
        "--ignore_out", default=False, action="store_true", help="ignore outside of view"
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    if args.ignore_out:
        ignore_dir = gt_dir.parent.joinpath("ignore_mask")

    output_dir.mkdir(exist_ok=True)

    pred_paths = list(Path(pred_dir).glob("*.png"))

    preds = []
    gts = []
    for pred_path in tqdm(pred_paths):
        img_name = pred_path.name
        gt_path = gt_dir.joinpath(img_name)

        pred = np.array(Image.open(pred_path).resize((1080, 1080), Image.NEAREST), dtype=np.int32)
        gt = np.array(Image.open(gt_path), dtype=np.int32)

        if args.ignore_out:
            ignore_path = ignore_dir.joinpath(img_name)
            ignore = np.array(Image.open(ignore_path), dtype=np.int32)
            gt[np.where(ignore == 1)] = -1

        preds.append(pred)
        gts.append(gt)

    results = eval_semantic_segmentation(preds, gts)
    pprint(results)

    results["class_accuracy"] = results["class_accuracy"].tolist()
    results["iou"] = results["iou"].tolist()
    if not args.ignore_out:
        with Path(args.output_dir).joinpath("semantic_segmentation_result.json").open("w") as f:
            json.dump(results, f, indent=4)
    else:
        with Path(args.output_dir).joinpath("ignore_semantic_segmentation_result.json").open(
            "w"
        ) as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
