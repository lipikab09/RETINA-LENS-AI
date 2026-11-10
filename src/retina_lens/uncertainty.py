"""Entropy, temperature scaling, MC-dropout, abstention, OOD scoring."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .models.multitask import grade_probs


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Fit T on VALIDATION data: p_cal ∝ p^(1/T) (works for CE and ordinal heads alike)."""
    logp = torch.log(torch.tensor(probs).clamp_min(1e-8)).float(); t = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100); yt = torch.tensor(y)
    def closure():
        opt.zero_grad(); loss = nn.functional.cross_entropy(logp / t.clamp_min(0.05), yt); loss.backward(); return loss
    opt.step(closure)
    return float(t.clamp_min(0.05).item())


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-8, 1)) / t; z -= z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


@torch.no_grad()
def mc_dropout_probs(model: nn.Module, x: torch.Tensor, mode: str, n: int = 10) -> torch.Tensor:
    """Returns [n,B,K] probabilities with ONLY dropout layers in train mode (BatchNorm stays in eval)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    out = torch.stack([grade_probs(model(x)["grade_logits"], mode) for _ in range(n)])
    model.eval()
    return out


def abstain(entropies: np.ndarray, threshold: float) -> np.ndarray:
    return entropies > threshold


def ood_auroc(in_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC of entropy as an OOD detector (higher entropy = more OOD)."""
    return float(roc_auc_score(np.r_[np.zeros(len(in_scores)), np.ones(len(ood_scores))], np.r_[in_scores, ood_scores]))
