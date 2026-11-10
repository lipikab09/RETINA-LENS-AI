"""Fundus preprocessing + heuristic image-quality metrics."""
from __future__ import annotations

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def fundus_bbox(img: np.ndarray, thr: int = 10) -> tuple[int, int, int, int]:
    """Bounding box (y0,y1,x0,x1) of the non-black fundus region."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > thr)
    if len(ys) == 0:
        return 0, img.shape[0], 0, img.shape[1]
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def resize_pad(arr: np.ndarray, size: int, mask: bool = False) -> np.ndarray:
    """Aspect-preserving resize + zero padding to size x size."""
    h, w = arr.shape[:2]; s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    if mask:
        r = (cv2.resize(arr.astype(np.uint8) * 255, (nw, nh), interpolation=cv2.INTER_AREA) >= 64).astype(np.uint8)
    else:
        r = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((size, size) + arr.shape[2:], dtype=r.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def apply_clahe(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def illumination_normalize(img: np.ndarray, sigma_frac: float = 0.04) -> np.ndarray:
    """Ben-Graham style local-mean subtraction (optional)."""
    s = max(1, int(img.shape[0] * sigma_frac))
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), s), -4, 128)


def fundus_mask(img: np.ndarray, thr: int = 10) -> np.ndarray:
    m = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > thr).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def quality_metrics(img: np.ndarray) -> dict[str, float]:
    """Heuristic blur / exposure / coverage measures on the fundus region."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); m = fundus_mask(img).astype(bool)
    if m.sum() == 0:
        return {"blur_var": 0.0, "brightness": 0.0, "fundus_fraction": 0.0, "overexposed_frac": 0.0}
    return {"blur_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "brightness": float(gray[m].mean()),
            "fundus_fraction": float(m.mean()), "overexposed_frac": float((gray[m] > 245).mean())}


def to_tensor_array(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 3xHxW ImageNet-normalised."""
    return np.ascontiguousarray(((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))


def preprocess_image(img: np.ndarray, size: int, clahe: bool = True) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Deterministic eval-time pipeline. Returns padded uint8 image and the crop bbox."""
    y0, y1, x0, x1 = fundus_bbox(img)
    out = resize_pad(img[y0:y1, x0:x1], size)
    return (apply_clahe(out) if clahe else out), (y0, y1, x0, x1)
