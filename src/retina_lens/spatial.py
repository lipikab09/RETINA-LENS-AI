"""Spatial lesion analysis + box derivation. Pixel units only: NOT calibrated clinical measurements."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .data.schema import LESIONS


def derive_boxes(masks: np.ndarray, prob: np.ndarray | None = None, min_area: int = 3) -> list[dict]:
    """Boxes DERIVED from segmentation components (not an independent detector; boxes of touching lesions merge)."""
    boxes = []
    for i, k in enumerate(LESIONS):
        lab, n = ndimage.label(masks[i])
        for j, sl in enumerate(ndimage.find_objects(lab), 1):
            comp = lab[sl] == j
            if comp.sum() < min_area:
                continue
            conf = float(prob[i][sl][comp].mean()) if prob is not None else 1.0
            boxes.append({"lesion": k, "x0": int(sl[1].start), "y0": int(sl[0].start), "x1": int(sl[1].stop), "y1": int(sl[0].stop),
                          "area_px": int(comp.sum()), "confidence": conf})
    return boxes


def analyze(masks: np.ndarray, fundus: np.ndarray, od_xy: tuple[float, float] | None = None,
            fovea_xy: tuple[float, float] | None = None, grid: int = 3) -> dict:
    """Counts, areas, centroids (normalised to fundus disc), concentric-zone/quadrant distribution, density maps."""
    ys, xs = np.where(fundus > 0)
    if len(ys) == 0:
        return {"lesions": [], "per_class": {}, "density": {}}
    cy, cx, R = ys.mean(), xs.mean(), max(np.ptp(ys), np.ptp(xs)) / 2 + 1e-6
    res: dict = {"lesions": [], "per_class": {}, "density": {}, "reference": {"center": [float(cx), float(cy)], "radius_px": float(R)},
                 "note": "Pixel/normalised units only; no calibrated anatomical measurement."}
    for i, k in enumerate(LESIONS):
        lab, n = ndimage.label(masks[i]); area = int(masks[i].sum())
        res["per_class"][k] = {"count": int(n), "area_px": area, "area_fraction_of_fundus": float(area / max(len(ys), 1))}
        if n:
            for j, (yc, xc) in enumerate(ndimage.center_of_mass(masks[i], lab, range(1, n + 1)), 1):
                r = float(np.hypot(xc - cx, yc - cy) / R)
                d = {"lesion": k, "x": float(xc), "y": float(yc), "r_norm": r,
                     "zone": "central" if r < 0.33 else "mid" if r < 0.66 else "peripheral",
                     "quadrant": ("lower" if yc > cy else "upper") + ("-right" if xc > cx else "-left"),  # image coordinates
                     "area_px": int((lab == j).sum())}
                if od_xy:
                    d["dist_to_optic_disc_norm"] = float(np.hypot(xc - od_xy[0], yc - od_xy[1]) / R)
                if fovea_xy:
                    d["dist_to_fovea_norm"] = float(np.hypot(xc - fovea_xy[0], yc - fovea_xy[1]) / R)
                res["lesions"].append(d)
        dens = ndimage.gaussian_filter(masks[i].astype(np.float32), sigma=max(masks[i].shape) / 40)
        res["density"][k] = ndimage.zoom(dens, 64 / dens.shape[0], order=1).tolist()
    return res
