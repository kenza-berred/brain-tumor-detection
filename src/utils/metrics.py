"""Detection metrics in NumPy: precision, recall, mAP@0.5 and mAP@[.5:.95].

Same definition as COCO / Ultralytics: a prediction is a true positive at IoU threshold t
when it overlaps a not-yet-matched ground-truth box of the same class with IoU >= t.
AP = area under the precision-recall curve (precision envelope, 101-point interpolation),
averaged over classes, and over the 10 IoU thresholds 0.50:0.05:0.95 for mAP50-95.
"""

from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (n, 4) and b (m, 4), format x1 y1 x2 y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_predictions(
    pred_boxes: np.ndarray, pred_cls: np.ndarray, gt_boxes: np.ndarray, gt_cls: np.ndarray
) -> np.ndarray:
    """(n_pred, 10) boolean matrix: is prediction i a true positive at each IoU threshold."""
    tp = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp
    iou = box_iou(gt_boxes, pred_boxes) * (gt_cls[:, None] == pred_cls[None, :])
    for k, t in enumerate(IOU_THRESHOLDS):
        gi, pi = np.nonzero(iou >= t)
        if not len(gi):
            continue
        order = np.argsort(-iou[gi, pi])  # best overlaps first, each box used once
        gi, pi = gi[order], pi[order]
        _, first_p = np.unique(pi, return_index=True)
        gi, pi = gi[first_p], pi[first_p]
        order = np.argsort(-iou[gi, pi])
        gi, pi = gi[order], pi[order]
        _, first_g = np.unique(gi, return_index=True)
        tp[pi[first_g], k] = True
    return tp


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the PR curve with the precision envelope and 101-point interpolation."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz
    return float(trapezoid(np.interp(x, mrec, mpre), x))


def detection_metrics(
    tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, gt_cls: np.ndarray, classes: list[int]
) -> dict[str, float]:
    """Aggregate per-image matches into precision, recall, mAP50 and mAP50-95 (mean over classes).

    Precision and recall are read at the confidence that maximises the mean F1 (like Ultralytics).
    """
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    grid = np.linspace(0, 1, 1000)
    ap = np.zeros((len(classes), len(IOU_THRESHOLDS)))
    p_curve = np.zeros((len(classes), len(grid)))
    r_curve = np.zeros((len(classes), len(grid)))
    for ci, c in enumerate(classes):
        sel = pred_cls == c
        n_gt = int((gt_cls == c).sum())
        if not sel.any() or n_gt == 0:
            continue
        tpc = tp[sel].cumsum(0)
        fpc = (1 - tp[sel]).cumsum(0)
        recall = tpc / (n_gt + 1e-16)
        precision = tpc / (tpc + fpc)
        # curves at IoU 0.5, as a function of confidence (conf decreases along the arrays)
        r_curve[ci] = np.interp(-grid, -conf[sel], recall[:, 0], left=0)
        p_curve[ci] = np.interp(-grid, -conf[sel], precision[:, 0], left=1)
        for k in range(len(IOU_THRESHOLDS)):
            ap[ci, k] = average_precision(recall[:, k], precision[:, k])
    f1 = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
    best = int(f1.mean(0).argmax())
    return {
        "precision": float(p_curve[:, best].mean()),
        "recall": float(r_curve[:, best].mean()),
        "mAP50": float(ap[:, 0].mean()),
        "mAP50_95": float(ap.mean()),
    }
