"""Aggregate runs/*/history.json (+metrics_test.json when present) into results/comparison.csv and a plot."""
import json
from pathlib import Path

import pandas as pd

rows = []
for h in sorted(Path("runs").glob("*/history.json")):
    hist = json.loads(h.read_text()); best = max(hist, key=lambda m: m.get("val_qwk", -1e9))
    row = {"run": h.parent.name, "best_epoch": best["epoch"], "val_qwk": best.get("val_qwk"), "val_dice": best.get("val_dice")}
    t = h.parent / "metrics_test.json"
    if t.exists():
        m = json.loads(t.read_text()); row.update(test_qwk=m["grading"]["qwk"], test_qwk_lo=m["qwk_ci95"][0], test_qwk_hi=m["qwk_ci95"][1], test_ece=m["calibration_temp_scaled"]["ece"])
    rows.append(row)
Path("results").mkdir(exist_ok=True); df = pd.DataFrame(rows); df.to_csv("results/comparison.csv", index=False); print(df.to_string(index=False))
