"""Single-image inference used by API and dashboard. Never returns dummy predictions."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .data.preprocess import fundus_mask, preprocess_image, quality_metrics, to_tensor_array
from .explain import grad_cam
from .models.multitask import build_model, grade_probs
from .spatial import analyze, derive_boxes
from .uncertainty import apply_temperature, entropy, mc_dropout_probs
from .utils import get_device


class CheckpointMissing(RuntimeError):
    """Raised when no trained checkpoint is available: inference is unavailable."""


class Predictor:
    def __init__(self, ckpt_path: str | Path, device: str = "auto"):
        p = Path(ckpt_path)
        if not p.exists():
            raise CheckpointMissing(f"Checkpoint not found: {p}")
        self.dev = get_device(device); ck = torch.load(p, map_location=self.dev, weights_only=False)
        self.cfg = ck["cfg"]; self.cfg["model"]["pretrained"] = False
        self.model = build_model(self.cfg).to(self.dev); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.mode = self.cfg["loss"]["grade"] if self.cfg["loss"]["grade"] in ("coral", "cumulative") else "ce"
        self.size = self.cfg["data"]["img_size"]; self.temperature = float(ck.get("temperature", 1.0))
        self.version = f"retina-lens {__version__} | {self.cfg['run_name']} | epoch {ck['epoch']}"
        self.synthetic = bool(self.cfg.get("synthetic"))

    def predict(self, rgb: np.ndarray, mc: bool = True, cam: bool = False, thr: float = 0.5) -> dict:
        t0 = time.perf_counter(); q = self.cfg["quality"]; u = self.cfg["uncertainty"]
        img, _ = preprocess_image(rgb, self.size, self.cfg["data"]["clahe"])
        x = torch.from_numpy(to_tensor_array(img))[None].to(self.dev)
        with torch.no_grad():
            out = self.model(x)
            probs = grade_probs(out["grade_logits"], self.mode)
            if mc:
                s = mc_dropout_probs(self.model, x, self.mode, u["mc_samples"]); probs = s.mean(0)
            seg = torch.sigmoid(out["seg_logits"])[0].cpu().numpy()
            qprob = float(torch.sigmoid(out["quality_logit"]).item())
        p = apply_temperature(probs.cpu().numpy(), self.temperature)[0]
        ent = float(entropy(p)); masks = seg > thr; fm = fundus_mask(img)
        masks = masks & (fm > 0)[None]
        qm = quality_metrics(img)
        adequate = (qm["blur_var"] >= q["min_blur_var"] and q["brightness_range"][0] <= qm["brightness"] <= q["brightness_range"][1]
                    and qm["fundus_fraction"] >= q["min_fundus_fraction"])
        res = {"model_version": self.version, "grade": int(p.argmax()), "probabilities": p.tolist(), "entropy": ent,
               "abstain": bool(ent > u["abstain_entropy"] or not adequate),
               "quality": {**qm, "heuristic_adequate": bool(adequate), "learned_quality_prob": qprob},
               "preprocessed": img, "masks": masks, "boxes": derive_boxes(masks, seg),
               "spatial": analyze(masks, fm), "seg_prob": seg, "temperature": self.temperature, "synthetic_model": self.synthetic}
        if cam:
            res["cam"] = grad_cam(self.model, x, mode_ordinal=self.mode != "ce")
        res["inference_ms"] = (time.perf_counter() - t0) * 1000
        return res
