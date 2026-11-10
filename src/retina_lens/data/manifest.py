"""Build the manifest + integrity checks.  python -m retina_lens.data.manifest --config configs/default.yaml"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from ..config import load_config
from ..utils import get_logger
from .adapters import ADAPTERS
from .schema import COLUMNS, LESIONS
from .splits import assign_splits

log = get_logger("manifest")


def build_manifest(cfg: dict) -> pd.DataFrame:
    dcfg = yaml.safe_load(Path(cfg["paths"]["datasets"]).read_text())
    recs: list[dict] = []
    for name in cfg["data"]["datasets"]:
        if name not in dcfg:
            log.warning("dataset %s missing in datasets.yaml; skipped", name); continue
        recs += ADAPTERS[name](dcfg[name])
    if not recs:
        raise RuntimeError("No records found. Check configs/datasets.yaml paths.")
    return assign_splits(pd.DataFrame(recs, columns=COLUMNS), cfg["seed"])


def integrity_report(df: pd.DataFrame) -> dict:
    rep = {"n": len(df), "duplicate_uid": int(df.uid.duplicated().sum()),
           "missing_files": int((~df.image_path.map(lambda p: Path(p).exists())).sum()),
           "per_dataset": df.groupby("dataset").size().to_dict(),
           "lesion_valid_counts": {k: int(df[f"valid_{k}"].astype(bool).sum()) for k in LESIONS}}
    pg = df.dropna(subset=["patient_id"]).groupby("patient_id").split.nunique()
    rep["patients_spanning_splits"] = int((pg > 1).sum())  # must be 0
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)
    df = build_manifest(cfg); out = Path(cfg["paths"]["manifest"]); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False); rep = integrity_report(df)
    log.info("manifest -> %s\n%s", out, yaml.safe_dump(rep, sort_keys=False))
    assert rep["duplicate_uid"] == 0 and rep["patients_spanning_splits"] == 0, "integrity check failed"


if __name__ == "__main__":
    main()
