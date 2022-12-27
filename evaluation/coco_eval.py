"""
The evaluate_objcd() and _summarize_objcd() are modified:
Copyright (c) 2017 Preferred Networks, Inc.
MIT License (see https://github.com/chainer/chainercv/blob/master/LICENSE)

Link:
    - https://github.com/chainer/chainercv/blob/dcce9fc53ec9e02623c847299aaeab5ac3443e2b/
      chainercv/evaluations/eval_detection_coco.py#L16-L273
    - https://github.com/chainer/chainercv/blob/dcce9fc53ec9e02623c847299aaeab5ac3443e2b/
      chainercv/evaluations/eval_detection_coco.py#L304-L333
"""

from typing import Tuple

import numpy as np


def evaluate_objcd(coco_eval):
    """
    Arguments:
        coco_eval (pycocotools.cocoeval.COCOeval)

    Returns:
        results (dict)
    """
    results = {}
    labels = sorted([category["id"] for category in coco_eval.cocoGt.dataset["categories"]])
    p = coco_eval.params
    common_kwargs = {
        "prec": coco_eval.eval["precision"],
        "rec": coco_eval.eval["recall"],
        "iou_threshs": p.iouThrs,
        "area_ranges": p.areaRngLbl,
        "max_detection_list": p.maxDets,
    }
    all_kwargs = {}
    for prec in [True, False]:
        for iou_thresh in [None, 0.50, 0.75]:
            for bbox_size in p.areaRngLbl:
                # make dict key
                ap_key = "precision" if prec else "recall"
                iou_key = "0.50:0.95" if iou_thresh is None else str(iou_thresh)
                key = "{}/iou={}/area={}".format(ap_key, iou_key, bbox_size)

                # add kwarg to dict
                all_kwargs[key] = {
                    "ap": prec,
                    "iou_thresh": iou_thresh,
                    "area_range": bbox_size,
                    "max_detection": 100,
                }

    for key, kwargs in all_kwargs.items():
        kwargs.update(common_kwargs)
        metrics, mean_metric = _summarize_objcd(**kwargs)

        results[key] = np.nan * np.ones(np.max(labels))
        results[key][: len(labels)] = metrics
        results["mean " + key] = mean_metric

    results["existent_labels"] = labels
    return results


def _summarize_objcd(
    prec,
    rec,
    iou_threshs,
    area_ranges,
    max_detection_list,
    ap=True,
    iou_thresh=None,
    area_range="all",
    max_detection=100,
) -> Tuple[np.array, np.array]:

    a_idx = area_ranges.index(area_range)
    m_idx = max_detection_list.index(max_detection)
    if ap:
        val_value = prec.copy()
        if iou_thresh is not None:
            val_value = val_value[iou_thresh == iou_threshs]
        val_value = val_value[:, :, a_idx, m_idx]
    else:
        val_value = rec.copy()
        if iou_thresh is not None:
            val_value = val_value[iou_thresh == iou_threshs]
        val_value = val_value[:, :, a_idx, m_idx]

    val_value[val_value == -1] = np.nan
    val_value = val_value.reshape((-1, val_value.shape[-1]))
    valid_classes = np.any(np.logical_not(np.isnan(val_value)), axis=0)
    cls_val_value = np.nan * np.ones(len(valid_classes), dtype=np.float32)
    cls_val_value[valid_classes] = np.nanmean(val_value[:, valid_classes], axis=0)

    if not np.any(valid_classes):
        mean_val_value = np.nan
    else:
        mean_val_value = np.nanmean(cls_val_value)
    return cls_val_value, mean_val_value
