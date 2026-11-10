"""Validity-masked multi-task losses. A task contributes only for samples whose labels exist."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def grade_loss(logits, y, valid, mode: str, class_w=None, gamma: float = 2.0) -> torch.Tensor:
    if valid.sum() == 0:
        return logits.sum() * 0.0
    lg, y = logits[valid], y[valid]
    if mode in ("coral", "cumulative"):
        levels = (y.unsqueeze(1) > torch.arange(lg.shape[1], device=y.device)).float()
        return F.binary_cross_entropy_with_logits(lg, levels)
    if mode == "wce":
        return F.cross_entropy(lg, y, weight=class_w)
    if mode == "focal":
        logp = F.log_softmax(lg, 1).gather(1, y[:, None])[:, 0]
        return (-(1 - logp.exp()) ** gamma * logp).mean()
    return F.cross_entropy(lg, y)


def seg_loss(logits, target, valid, pos_weight=None) -> torch.Tensor:
    """BCE(+pos_weight) + soft Dice per (sample, lesion); averaged only over valid pairs."""
    if valid.sum() == 0:
        return logits.sum() * 0.0
    pw = None if pos_weight is None else pos_weight.view(1, -1, 1, 1).to(logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none").mean((2, 3))
    p = torch.sigmoid(logits)
    inter, den = (p * target).sum((2, 3)), p.sum((2, 3)) + target.sum((2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    return ((bce + dice) * valid.float()).sum() / valid.float().sum()


def quality_loss(logit, y, valid) -> torch.Tensor:
    if valid.sum() == 0:
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[valid], y[valid])


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: dict, class_w=None):
        super().__init__()
        self.c, self.class_w = cfg["loss"], class_w
        self.pw = torch.tensor(self.c["seg_pos_weight"])
        self.uncertainty = self.c["weighting"] == "uncertainty"
        self.log_vars = nn.Parameter(torch.zeros(3))  # Kendall et al. homoscedastic weighting (grade, seg, quality)

    def forward(self, out: dict, b: dict) -> tuple[torch.Tensor, dict]:
        lg = grade_loss(out["grade_logits"], b["grade"], b["grade_valid"], self.c["grade"], self.class_w, self.c["focal_gamma"])
        ls = 0.5 * (seg_loss(out["seg_logits"], b["masks"], b["mask_valid"], self.pw)
                    + seg_loss(out["aux_seg_logits"], b["masks"], b["mask_valid"], self.pw))
        lq = quality_loss(out["quality_logit"], b["quality"], b["quality_valid"])
        parts = torch.stack([lg, ls, lq]); lam = torch.tensor([self.c["lambda_grade"], self.c["lambda_seg"], self.c["lambda_quality"]], device=parts.device)
        active = torch.stack([b["grade_valid"].any(), b["mask_valid"].any(), b["quality_valid"].any()]).float() * (lam > 0).float()
        if self.uncertainty:
            total = (active * (torch.exp(-self.log_vars) * parts + self.log_vars)).sum()
        else:
            total = (active * lam * parts).sum()
        return total, {"grade": lg.item(), "seg": ls.item(), "quality": lq.item(), "total": total.item()}
