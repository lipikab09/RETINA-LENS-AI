"""Evaluation metrics: grading, calibration, segmentation, bootstrap CIs."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import (accuracy_score, brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def qwk(y, pred) -> float:
    return float(cohen_kappa_score(y, pred, weights="quadratic"))


def grading_metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1); k = probs.shape[1]; labels = list(range(k))
    cm = confusion_matrix(y, pred, labels=labels)
    per_auc, sens, spec = {}, {}, {}
    for c in labels:
        yb = (y == c).astype(int)
        per_auc[c] = float(roc_auc_score(yb, probs[:, c])) if 0 < yb.sum() < len(yb) else None
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
        sens[c] = float(tp / (tp + fn)) if tp + fn else None
        spec[c] = float(tn / (tn + fp)) if tn + fp else None
    return {"accuracy": float(accuracy_score(y, pred)), "qwk": qwk(y, pred),
            "macro_precision": float(precision_score(y, pred, average="macro", labels=labels, zero_division=0)),
            "macro_recall": float(recall_score(y, pred, average="macro", labels=labels, zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
            "auroc_per_class": per_auc, "sensitivity": sens, "specificity": spec, "confusion_matrix": cm.tolist(),
            "class_counts": np.bincount(y, minlength=k).tolist()}


def calibration(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> dict:
    conf, pred = probs.max(1), probs.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); ece, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean()); rel.append((float(conf[m].mean()), float(acc[m].mean()), int(m.sum())))
    onehot = np.eye(probs.shape[1])[y]
    return {"ece": float(ece), "brier": float(((probs - onehot) ** 2).sum(1).mean()), "reliability": rel}


def bootstrap_ci(fn, y, probs, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        try:
            vals.append(fn(y[i], probs[i]))
        except ValueError:
            continue
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    inter = (pred & gt).sum(); return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps)), float((inter + eps) / ((pred | gt).sum() + eps))


def lesion_wise(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Lesion-wise sensitivity (GT components overlapped by prediction) and false-positive components."""
    gl, gn = ndimage.label(gt); pl, pn = ndimage.label(pred)
    hit = sum(1 for i in range(1, gn + 1) if pred[gl == i].any())
    fp = sum(1 for j in range(1, pn + 1) if not gt[pl == j].any())
    return {"gt_lesions": gn, "detected": hit, "false_positives": fp}
