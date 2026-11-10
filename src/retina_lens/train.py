"""Training:  python -m retina_lens.train --config configs/default.yaml [--resume runs/<name>/last.pt]"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import flatten, load_config
from .data.dataset import RetinaDataset, SyntheticRetinaDataset
from .losses import MultiTaskLoss
from .metrics import dice_iou, qwk
from .models.multitask import build_model, grade_probs
from .utils import get_device, get_logger, seed_everything

log = get_logger("train")


def make_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, torch.Tensor | None]:
    d = cfg["data"]; bs, nw, size = d["batch_size"], d["num_workers"], d["img_size"]
    if cfg.get("synthetic"):  # smoke-test only
        tr, va, cw = SyntheticRetinaDataset(16, size, 0), SyntheticRetinaDataset(8, size, 1000), None
        return DataLoader(tr, bs, shuffle=True), DataLoader(va, bs), cw
    df = pd.read_csv(cfg["paths"]["manifest"])
    trdf, vadf = df[df.split == "train"], df[df.split == "val"]          # TEST split is never touched here
    gv = trdf[trdf.grade_valid.astype(bool)].grade.astype(int)
    counts = np.bincount(gv, minlength=5).astype(float) + 1
    cw = torch.tensor(counts.sum() / (5 * counts), dtype=torch.float32) if d["imbalance"] == "weighted_loss" else None
    sampler = None
    if d["imbalance"] == "sampler":
        w = np.where(trdf.grade_valid.astype(bool), (1.0 / counts)[trdf.grade.clip(lower=0).astype(int)], float(np.median(1.0 / counts)))
        sampler = WeightedRandomSampler(w, len(trdf), replacement=True)
    mk = lambda df_, train: RetinaDataset(df_, size, train, d["clahe"])
    return (DataLoader(mk(trdf, True), bs, shuffle=sampler is None, sampler=sampler, num_workers=nw, drop_last=True, pin_memory=True),
            DataLoader(mk(vadf, False), bs, num_workers=nw, pin_memory=True), cw)


def to_dev(b: dict, dev) -> dict:
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def validate(model, crit, loader, dev, mode: str) -> dict:
    model.eval(); ys, ps, losses, dices = [], [], [], []
    for b in loader:
        b = to_dev(b, dev); out = model(b["image"]); losses.append(crit(out, b)[1]["total"])
        v = b["grade_valid"]
        if v.any():
            ys.append(b["grade"][v].cpu().numpy()); ps.append(grade_probs(out["grade_logits"], mode)[v].cpu().numpy())
        pred = (torch.sigmoid(out["seg_logits"]) > 0.5).cpu().numpy(); gt = b["masks"].cpu().numpy() > 0.5; mv = b["mask_valid"].cpu().numpy()
        for i in range(len(pred)):
            for c in range(pred.shape[1]):
                if mv[i, c] and gt[i, c].any():
                    dices.append(dice_iou(pred[i, c], gt[i, c])[0])
    res = {"val_loss": float(np.mean(losses)), "val_dice": float(np.mean(dices)) if dices else float("nan")}
    if ys:
        y, p = np.concatenate(ys), np.concatenate(ps); res["val_qwk"] = qwk(y, p.argmax(1)); res["val_acc"] = float((y == p.argmax(1)).mean())
    return res


def lr_at(step: int, total: int, warm: int, base: float) -> float:
    return base * step / max(1, warm) if step < warm else base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def main(cfg: dict, resume: str | None = None) -> Path:
    seed_everything(cfg["seed"]); dev = get_device(cfg["device"]); t = cfg["train"]
    run_dir = Path(cfg["paths"]["runs"]) / cfg["run_name"]; run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    tr, va, cw = make_loaders(cfg); model = build_model(cfg).to(dev)
    crit = MultiTaskLoss(cfg, cw.to(dev) if cw is not None else None).to(dev)
    params = list(model.parameters()) + list(crit.parameters())
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    use_amp = bool(t["amp"]) and dev.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mode = cfg["loss"]["grade"] if cfg["loss"]["grade"] in ("coral", "cumulative") else "ce"
    start, best, bad, hist = 0, -1e9, 0, []
    if resume and Path(resume).exists():
        ck = torch.load(resume, map_location=dev); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best, hist = ck["epoch"] + 1, ck["best"], ck.get("hist", []); log.info("resumed from epoch %d", start)
    try:
        import mlflow; mlflow.set_experiment("retina-lens"); mlflow.start_run(run_name=cfg["run_name"]); mlflow.log_params(flatten(cfg)); ml = mlflow
    except Exception:
        ml = None
    total = t["epochs"] * len(tr); step = start * len(tr)
    for ep in range(start, t["epochs"]):
        for p in model.encoder.parameters():
            p.requires_grad = ep >= t["freeze_backbone_epochs"]
        model.train(); run = []
        for b in tqdm(tr, desc=f"epoch {ep}", leave=False):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, t["warmup_epochs"] * len(tr), t["lr"])
            b = to_dev(b, dev)
            with torch.autocast(dev.type, enabled=use_amp):
                loss, parts = crit(model(b["image"]), b)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip"]); scaler.step(opt); scaler.update(); run.append(parts["total"]); step += 1
        m = {"epoch": ep, "train_loss": float(np.mean(run)), **validate(model, crit, va, dev, mode)}
        hist.append(m); log.info(m)
        if ml:
            ml.log_metrics({k: v for k, v in m.items() if isinstance(v, float) and not math.isnan(v)}, step=ep)
        score = m.get("val_qwk", -m["val_loss"])
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": max(best, score), "hist": hist, "cfg": cfg}
        torch.save(state, run_dir / "last.pt")
        if score > best:
            best, bad = score, 0; torch.save(state, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= t["patience"]:
                log.info("early stopping"); break
    (run_dir / "history.json").write_text(json.dumps(hist, indent=2))
    if ml:
        ml.end_run()
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--resume")
    a = ap.parse_args(); main(load_config(a.config), a.resume)
