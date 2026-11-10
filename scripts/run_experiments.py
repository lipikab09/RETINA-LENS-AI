"""Experiment grid. Each entry = config overrides. Run: python scripts/run_experiments.py --exp exp1 [--debug]
Results land in runs/<name>/ and MLflow; compare with scripts/compare.py. GradNorm is NOT implemented (documented)."""
import argparse

from retina_lens.config import deep_update, load_config
from retina_lens.train import main as train_main

B = lambda **kw: kw
EXPERIMENTS = {
    "exp1_backbone": {f"bb_{n}": B(model={"backbone": n}) for n in ["tf_efficientnetv2_s", "convnext_tiny", "swin_tiny_patch4_window7_224"]},
    "exp2_ordinal": {f"loss_{m}": B(loss={"grade": m}) for m in ["ce", "wce", "focal", "coral", "cumulative"]},
    "exp3_single_vs_multi": {"single_task": B(loss={"lambda_seg": 0.0, "lambda_quality": 0.0}, model={"lesion_attention": False}), "multi_task": B()},
    "exp5_attention": {"attn_on": B(model={"lesion_attention": True}), "attn_off": B(model={"lesion_attention": False})},
    "exp7_imbalance": {f"imb_{m}": B(data={"imbalance": m}) for m in ["none", "weighted_loss", "sampler"]},
    "exp10_weighting": {"w_fixed": B(loss={"weighting": "fixed"}), "w_uncertainty": B(loss={"weighting": "uncertainty"})},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True, choices=EXPERIMENTS); ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(); base = load_config("configs/debug_cpu.yaml" if a.debug else "configs/default.yaml")
    for name, ov in EXPERIMENTS[a.exp].items():
        cfg = deep_update(base, ov); cfg["run_name"] = f"{a.exp}__{name}"; train_main(cfg)
