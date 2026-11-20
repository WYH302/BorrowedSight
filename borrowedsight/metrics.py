"""Metrics computed from a real prediction. Missing classes are skipped, not scored as 1."""

from __future__ import annotations

import numpy as np

from borrowedsight.labels import EMPTY, IGNORE_INDEX, N_CLASS, ROAD, VEHICLE

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


def semantic_target(semantic: np.ndarray, tristate: np.ndarray) -> np.ndarray:
    """Unknown and occupied-unlabeled use ignore_index. Only observed-free is empty."""
    sem = np.asarray(semantic)
    tri = np.asarray(tristate)
    target = np.full(sem.shape, IGNORE_INDEX, dtype=np.int64)
    target[tri == FREE] = EMPTY
    occupied = tri == OCCUPIED
    known = occupied & (sem >= 1) & (sem <= 12)
    target[known] = sem[known]
    return target


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num) / float(den)


def binary_f1(pred_positive: np.ndarray, gt_positive: np.ndarray) -> float | None:
    pred = np.asarray(pred_positive, dtype=bool)
    gt = np.asarray(gt_positive, dtype=bool)
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    if tp + fn == 0:
        return None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_prediction(pred_sem, pred_tri, gt_sem, gt_tri) -> dict:
    pred_sem = np.asarray(pred_sem)
    pred_tri = np.asarray(pred_tri)
    gt_sem = np.asarray(gt_sem)
    gt_tri = np.asarray(gt_tri)
    gt_target = semantic_target(gt_sem, gt_tri)
    region = gt_target != IGNORE_INDEX
    claims = pred_tri != UNKNOWN

    ious = []
    per_class = {}
    for cls in range(N_CLASS):
        gt_c = region & (gt_target == cls)
        if int(gt_c.sum()) == 0:
            continue
        pred_c = claims & region & (pred_sem == cls)
        inter = int(np.count_nonzero(gt_c & pred_c))
        union = int(np.count_nonzero(gt_c | pred_c))
        iou = inter / union if union else 0.0
        ious.append(iou)
        per_class[cls] = float(iou)

    unk = gt_tri == UNKNOWN
    ufe = _safe_div(
        float(np.count_nonzero((pred_tri == FREE) & unk)),
        float(np.count_nonzero(unk)),
    )
    return {
        "mIoU": float(np.mean(ious)) if ious else None,
        "vehicle_IoU": per_class.get(VEHICLE),
        "road_IoU": per_class.get(ROAD),
        "unknown_F1": binary_f1(pred_tri == UNKNOWN, unk),
        "unobserved_as_free_err": ufe,
        "n_classes_present": len(ious),
        "per_class_iou": per_class,
    }


def new_counts() -> dict:
    return {
        "inter": np.zeros(N_CLASS, dtype=np.float64),
        "pred": np.zeros(N_CLASS, dtype=np.float64),
        "gt": np.zeros(N_CLASS, dtype=np.float64),
        "ufe_num": 0.0,
        "ufe_den": 0.0,
        "unk_tp": 0.0,
        "unk_fp": 0.0,
        "unk_fn": 0.0,
    }


def update_counts(acc: dict, pred_sem, pred_tri, gt_sem, gt_tri) -> None:
    pred_sem = np.asarray(pred_sem)
    pred_tri = np.asarray(pred_tri)
    gt_sem = np.asarray(gt_sem)
    gt_tri = np.asarray(gt_tri)
    gt_target = semantic_target(gt_sem, gt_tri)
    region = gt_target != IGNORE_INDEX
    claims = pred_tri != UNKNOWN
    for cls in range(N_CLASS):
        gt_c = region & (gt_target == cls)
        pred_c = claims & region & (pred_sem == cls)
        acc["gt"][cls] += float(np.count_nonzero(gt_c))
        acc["pred"][cls] += float(np.count_nonzero(pred_c))
        acc["inter"][cls] += float(np.count_nonzero(gt_c & pred_c))
    unk = gt_tri == UNKNOWN
    pred_unk = pred_tri == UNKNOWN
    acc["ufe_num"] += float(np.count_nonzero((pred_tri == FREE) & unk))
    acc["ufe_den"] += float(np.count_nonzero(unk))
    acc["unk_tp"] += float(np.count_nonzero(pred_unk & unk))
    acc["unk_fp"] += float(np.count_nonzero(pred_unk & ~unk))
    acc["unk_fn"] += float(np.count_nonzero(~pred_unk & unk))


def finalize_counts(acc: dict) -> dict:
    ious = []
    per_class = {}
    for cls in range(N_CLASS):
        if acc["gt"][cls] <= 0:
            continue
        union = acc["gt"][cls] + acc["pred"][cls] - acc["inter"][cls]
        iou = acc["inter"][cls] / union if union > 0 else 0.0
        ious.append(iou)
        per_class[cls] = float(iou)
    if acc["unk_tp"] + acc["unk_fn"] == 0:
        unknown_f1 = None
    else:
        precision = acc["unk_tp"] / (acc["unk_tp"] + acc["unk_fp"]) if (acc["unk_tp"] + acc["unk_fp"]) else 0.0
        recall = acc["unk_tp"] / (acc["unk_tp"] + acc["unk_fn"])
        unknown_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    ufe = None if acc["ufe_den"] <= 0 else acc["ufe_num"] / acc["ufe_den"]
    return {
        "mIoU": float(np.mean(ious)) if ious else None,
        "vehicle_IoU": per_class.get(VEHICLE),
        "road_IoU": per_class.get(ROAD),
        "unknown_F1": None if unknown_f1 is None else float(unknown_f1),
        "unobserved_as_free_err": None if ufe is None else float(ufe),
        "n_classes_present": len(ious),
        "per_class_iou": per_class,
    }


def mbps_from_bytes(bytes_per_frame: float, hz: float = 10.0) -> float:
    return float(bytes_per_frame) * float(hz) * 8.0 / 1.0e6
