"""Unified annotation schema. Missing annotations are *unknown* (valid_*=False), never negative."""
from __future__ import annotations

LESIONS = ["MA", "HE", "EX", "SE"]
LESION_NAMES = {"MA": "Microaneurysms", "HE": "Hemorrhages", "EX": "Hard exudates", "SE": "Soft exudates"}
GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

COLUMNS = (
    ["uid", "dataset", "image_id", "image_path", "patient_id", "width", "height",
     "grade", "grade_valid", "quality", "quality_valid", "split"]
    + [f"mask_{k}" for k in LESIONS] + [f"valid_{k}" for k in LESIONS]
)


def make_record(dataset: str, image_id: str, image_path: str, width: int, height: int, **kw) -> dict:
    """Create a record with every field defaulting to 'not available'."""
    rec = {c: None for c in COLUMNS}
    rec.update(uid=f"{dataset}:{image_id}", dataset=dataset, image_id=image_id, image_path=image_path,
               patient_id=kw.get("patient_id"), width=width, height=height,
               grade=-1, grade_valid=False, quality=-1, quality_valid=False, split="")
    for k in LESIONS:
        rec[f"mask_{k}"] = ""
        rec[f"valid_{k}"] = False
    rec.update({k: v for k, v in kw.items() if k in rec})
    return rec
