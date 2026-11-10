"""Grad-CAM (attribution) and overlays. Attribution heatmaps are NOT lesion ground truth."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .data.schema import LESIONS

COLORS = {"MA": (255, 215, 0), "HE": (230, 40, 40), "EX": (255, 255, 255), "SE": (60, 200, 255)}


def grad_cam(model, x: torch.Tensor, class_idx: int | None = None, mode_ordinal: bool = False) -> np.ndarray:
    """Grad-CAM on the last encoder stage for a single image x [1,3,H,W]. Returns HxW in [0,1]."""
    model.eval(); x = x.clone().requires_grad_(True)
    out = model(x); lg = out["grade_logits"]
    score = lg.sum() if mode_ordinal else lg[0, int(lg.argmax(1)) if class_idx is None else class_idx]
    fmap = model.last_feat
    grads = torch.autograd.grad(score, fmap, retain_graph=False)[0]
    w = grads.mean((2, 3), keepdim=True); cam = torch.relu((w * fmap).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min(); cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def overlay(img: np.ndarray, masks: np.ndarray | None = None, show: dict[str, bool] | None = None,
            heat: np.ndarray | None = None, boxes: list[dict] | None = None, alpha: float = 0.55) -> np.ndarray:
    """img uint8 HxWx3; masks [4,H,W] bool; toggles per lesion; optional heatmap and boxes."""
    out = img.copy(); show = show or {k: True for k in LESIONS}
    if heat is not None:
        hm = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        out = cv2.addWeighted(out, 1 - 0.4, hm, 0.4, 0)
    if masks is not None:
        for i, k in enumerate(LESIONS):
            if show.get(k, False) and masks[i].any():
                col = np.array(COLORS[k], np.uint8); m = masks[i].astype(bool)
                out[m] = (alpha * col + (1 - alpha) * out[m]).astype(np.uint8)
    for b in boxes or []:
        if show.get(b["lesion"], False):
            cv2.rectangle(out, (b["x0"], b["y0"]), (b["x1"], b["y1"]), COLORS[b["lesion"]], 1)
    return out
