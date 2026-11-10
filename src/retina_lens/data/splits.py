"""Reproducible, leakage-aware splits (grouped by patient_id when known, else by image)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_splits(df: pd.DataFrame, seed: int = 42, n_folds: int = 10) -> pd.DataFrame:
    """~80/10/10 train/val/test per dataset (official DDR splits are not used; document if you prefer them)."""
    df = df.copy(); df["split"] = "train"
    for _, sub in df.groupby("dataset"):
        groups = sub["patient_id"].fillna(sub["image_id"]).astype(str).values
        y = np.where(sub["grade_valid"].astype(bool), sub["grade"], 0)
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(skf.split(sub, y, groups))
        df.loc[sub.index[folds[0][1]], "split"] = "test"
        df.loc[sub.index[folds[1][1]], "split"] = "val"
    return df
