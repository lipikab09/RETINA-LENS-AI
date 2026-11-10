"""Test-set evaluation with temperature fit on VAL.  python -m retina_lens.evaluate --run runs/<name>"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .metrics import bootstrap_ci, calibration, grading_metrics, qwk
from .models.multitask import build_model, grade_probs
from .train import make_loaders, to_dev
from .uncertainty import apply_temperature, fit_temperature
from .utils import get_device


@torch.no_grad()
def collect(model, loader, dev, mode):
    ys, ps = [], []
    model.eval()
    for b in loader:
        b = to_dev(b, dev); v = b["grade_valid"]
        if v.any():
            ys.append(b["grade"][v].cpu().numpy()); ps.append(grade_probs(model(b["image"])["grade_logits"], mode)[v].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main(run: str, split: str = "test", n_boot: int = 1000) -> dict:
    ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False); cfg = ck["cfg"]; cfg["model"]["pretrained"] = False
    dev = get_device(cfg["device"]); model = build_model(cfg).to(dev); model.load_state_dict(ck["model"])
    mode = cfg["loss"]["grade"] if cfg["loss"]["grade"] in ("coral", "cumulative") else "ce"
    _, val_loader, _ = make_loaders(cfg)
    yv, pv = collect(model, val_loader, dev, mode); T = fit_temperature(pv, yv)      # fit on validation only
    import pandas as pd
    from .data.dataset import RetinaDataset
    from torch.utils.data import DataLoader
    df = pd.read_csv(cfg["paths"]["manifest"]); sub = df[df.split == split]
    loader = DataLoader(RetinaDataset(sub, cfg["data"]["img_size"], False, cfg["data"]["clahe"]), cfg["data"]["batch_size"])
    y, p = collect(model, loader, dev, mode); pc = apply_temperature(p, T)
    res = {"split": split, "n": int(len(y)), "temperature": T, "grading": grading_metrics(y, p),
           "qwk_ci95": bootstrap_ci(lambda a, b: qwk(a, b.argmax(1)), y, p, n_boot),
           "calibration_raw": calibration(y, p), "calibration_temp_scaled": calibration(y, pc),
           "disclaimer": "Research benchmark only. Not clinically validated."}
    ck["temperature"] = T; torch.save(ck, Path(run) / "best.pt")
    out = Path(run) / f"metrics_{split}.json"; out.write_text(json.dumps(res, indent=2)); return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--split", default="test")
    a = ap.parse_args(); r = main(a.run, a.split); print(json.dumps({k: r[k] for k in ("n", "qwk_ci95", "temperature")}, indent=2))
