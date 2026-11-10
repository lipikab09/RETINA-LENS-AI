"""Torch datasets with per-sample annotation-validity flags."""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import apply_clahe, fundus_bbox, resize_pad, to_tensor_array
from .schema import LESIONS


def build_augment(train: bool) -> A.Compose | None:
    if not train:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-30, 30), translate_percent=(-0.05, 0.05), p=0.7),  # identical warp on masks
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.HueSaturationValue(5, 10, 5, p=0.3),  # mild colour jitter only
    ])


def _read_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m.max(-1)
    return (m > 0).astype(np.uint8)


class RetinaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, train: bool, clahe: bool = True):
        self.df, self.size, self.clahe, self.aug = df.reset_index(drop=True), img_size, clahe, build_augment(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(r.image_path), cv2.COLOR_BGR2RGB)
        y0, y1, x0, x1 = fundus_bbox(img)
        img = resize_pad(img[y0:y1, x0:x1], self.size)
        masks, valid = [], []
        for k in LESIONS:
            ok, p = bool(r[f"valid_{k}"]), r[f"mask_{k}"]
            if ok and isinstance(p, str) and p:
                masks.append(resize_pad(_read_mask(p)[y0:y1, x0:x1], self.size, mask=True))
            else:  # asserted-negative or unknown -> zeros; the validity flag tells the loss which it is
                masks.append(np.zeros((self.size, self.size), np.uint8))
            valid.append(ok)
        if self.clahe:
            img = apply_clahe(img)
        if self.aug is not None:
            o = self.aug(image=img, masks=masks); img, masks = o["image"], o["masks"]
        return {
            "image": torch.from_numpy(to_tensor_array(img)),
            "masks": torch.from_numpy(np.stack(masks).astype(np.float32)),
            "mask_valid": torch.tensor(valid, dtype=torch.bool),
            "grade": torch.tensor(int(r.grade), dtype=torch.long),
            "grade_valid": torch.tensor(bool(r.grade_valid)),
            "quality": torch.tensor(float(max(r.quality, 0))),
            "quality_valid": torch.tensor(bool(r.quality_valid)),
        }


class SyntheticRetinaDataset(Dataset):
    """Procedural fake fundus images for CPU smoke tests ONLY. Never use for results."""

    def __init__(self, n: int, img_size: int, seed: int = 0):
        self.n, self.size, self.seed = n, img_size, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng(self.seed + i); s = self.size
        yy, xx = np.mgrid[:s, :s]; disc = ((yy - s / 2) ** 2 + (xx - s / 2) ** 2) < (s / 2.1) ** 2
        img = np.zeros((s, s, 3), np.float32); img[disc] = [0.55, 0.25, 0.1]
        grade = int(rng.integers(0, 5)); masks = np.zeros((4, s, s), np.float32)
        for c in range(4):
            for _ in range(int(rng.integers(0, 1 + grade))):
                cy, cx = rng.integers(s // 4, 3 * s // 4, 2); r = int(rng.integers(1, 4))
                blob = (((yy - cy) ** 2 + (xx - cx) ** 2) < r * r) & disc
                masks[c][blob] = 1; img[blob] = [0.9, 0.8, 0.2] if c >= 2 else [0.4, 0.05, 0.05]
        x = (img - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return {"image": torch.from_numpy(x.transpose(2, 0, 1).copy()), "masks": torch.from_numpy(masks),
                "mask_valid": torch.ones(4, dtype=torch.bool), "grade": torch.tensor(grade),
                "grade_valid": torch.tensor(True), "quality": torch.tensor(1.0), "quality_valid": torch.tensor(True)}
