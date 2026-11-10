"""Per-dataset adapters -> unified records. Layouts follow the public releases; verify against YOUR copy
(configure paths in configs/datasets.yaml). Any annotation we cannot find stays *invalid* (unknown)."""
from __future__ import annotations

import csv
import glob
from pathlib import Path

from PIL import Image

from ..utils import get_logger
from .schema import LESIONS, make_record

log = get_logger("adapters")


def _size(p: Path) -> tuple[int, int]:
    with Image.open(p) as im:  # reads header only
        return im.size


def _set_mask(rec: dict, key: str, path: Path, missing_is_negative: bool) -> None:
    if path.exists():
        rec[f"mask_{key}"], rec[f"valid_{key}"] = str(path), True
    elif missing_is_negative:  # only when the user explicitly asserts this protocol
        rec[f"mask_{key}"], rec[f"valid_{key}"] = "", True


def load_idrid(cfg: dict) -> list[dict]:
    root = Path(cfg["root"]); neg = cfg.get("missing_mask_means_negative", False)
    grades: dict[str, int] = {}
    for f in glob.glob(str(root / "**" / "*Disease Grading*.csv"), recursive=True):
        with open(f, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                grades[row["Image name"].strip()] = int(row["Retinopathy grade"])
    folders = {"MA": ("1. Microaneurysms", "MA"), "HE": ("2. Haemorrhages", "HE"),
               "EX": ("3. Hard Exudates", "EX"), "SE": ("4. Soft Exudates", "SE")}
    out = []
    for subset in ("a. Training Set", "b. Testing Set"):
        for img in sorted((root / "1. Original Images" / subset).glob("*.jpg")):
            w, h = _size(img)
            g = grades.get(img.stem)
            rec = make_record("idrid", img.stem, str(img), w, h, patient_id=img.stem,
                              grade=-1 if g is None else g, grade_valid=g is not None)
            for k, (folder, suf) in folders.items():
                _set_mask(rec, k, root / "2. All Segmentation Groundtruths" / subset / folder / f"{img.stem}_{suf}.tif", neg)
            out.append(rec)
    log.info("IDRiD: %d images", len(out))
    return out


def load_aptos(cfg: dict) -> list[dict]:
    root = Path(cfg["root"]); out = []
    with open(root / "train.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            img = root / "train_images" / f"{row['id_code']}.png"
            if not img.exists():
                continue
            w, h = _size(img)
            out.append(make_record("aptos", row["id_code"], str(img), w, h, grade=int(row["diagnosis"]), grade_valid=True))
    log.info("APTOS: %d images (labels only; no lesion masks)", len(out))
    return out


def load_ddr(cfg: dict) -> list[dict]:
    """DDR: grade 5 = ungradable -> grade invalid, quality=0. Segmentation subset is separate."""
    root = Path(cfg["root"]); neg = cfg.get("missing_mask_means_negative", False); out = []
    for split in ("train", "valid", "test"):
        txt = root / "DR_grading" / f"{split}.txt"
        if txt.exists():
            for line in txt.read_text().splitlines():
                parts = line.split()
                if len(parts) != 2:
                    continue
                img = root / "DR_grading" / split / parts[0]
                if not img.exists():
                    continue
                w, h = _size(img); g = int(parts[1]); ungr = g == 5
                out.append(make_record("ddr", img.stem, str(img), w, h, grade=-1 if ungr else g,
                                       grade_valid=not ungr, quality=0 if ungr else 1, quality_valid=True))
        seg_dir = root / "lesion_segmentation" / split
        for img in sorted((seg_dir / "image").glob("*.jpg")):
            w, h = _size(img)
            rec = make_record("ddr", f"seg_{img.stem}", str(img), w, h)
            for k in LESIONS:
                _set_mask(rec, k, seg_dir / "label" / k / f"{img.stem}.tif", neg)
            out.append(rec)
    log.info("DDR: %d records", len(out))
    return out


def load_fgadr(cfg: dict) -> list[dict]:
    root = Path(cfg["root"]) / "Seg-set"; neg = cfg.get("missing_mask_means_negative", False)
    grades: dict[str, int] = {}
    gcsv = root / "DR_Seg_Grading_Label.csv"
    if gcsv.exists():
        for row in csv.reader(open(gcsv, encoding="utf-8-sig")):
            if len(row) >= 2 and row[1].strip().isdigit():
                grades[row[0].strip()] = int(row[1])
    out = []
    for img in sorted((root / "Original_Images").glob("*.png")):
        w, h = _size(img); g = grades.get(img.name)
        rec = make_record("fgadr", img.stem, str(img), w, h, grade=-1 if g is None else g, grade_valid=g is not None)
        for k in LESIONS:
            _set_mask(rec, k, root / f"{k}_Masks" / img.name, neg)
        out.append(rec)
    log.info("FGADR: %d images", len(out))
    return out


ADAPTERS = {"idrid": load_idrid, "aptos": load_aptos, "ddr": load_ddr, "fgadr": load_fgadr}
