"""PDF report generation (research output, not a diagnosis)."""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .data.schema import GRADE_NAMES, LESION_NAMES, LESIONS
from .explain import overlay

DISCLAIMER = "RESEARCH PREDICTION - NOT A MEDICAL DIAGNOSIS. Not clinically validated; do not use for patient care."


def build_pdf(image_id: str, res: dict) -> bytes:
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); W, H = A4; y = H - 50
    c.setFont("Helvetica-Bold", 16); c.drawString(40, y, "RETINA-LENS AI - Analysis Report"); y -= 18
    c.setFont("Helvetica", 8); c.setFillColorRGB(0.7, 0.1, 0.1); c.drawString(40, y, DISCLAIMER); c.setFillColorRGB(0, 0, 0); y -= 18
    c.setFont("Helvetica", 10)
    for line in [f"Image: {image_id}", f"Model: {res['model_version']}", f"Generated: {datetime.utcnow():%Y-%m-%d %H:%M} UTC",
                 f"Predicted severity: {res['grade']} ({GRADE_NAMES[res['grade']]})  | entropy {res['entropy']:.3f}  | abstain flag: {res['abstain']}",
                 f"Image quality (heuristic adequate): {res['quality']['heuristic_adequate']} | blur var {res['quality']['blur_var']:.1f} | brightness {res['quality']['brightness']:.0f}"]:
        c.drawString(40, y, line); y -= 14
    y -= 4
    for i, p in enumerate(res["probabilities"]):
        c.drawString(50, y, f"{GRADE_NAMES[i]}: {p:.3f}"); y -= 12
    y -= 6; c.drawString(40, y, "Lesion summary (pixel units; boxes derived from segmentation):"); y -= 14
    for k in LESIONS:
        s = res["spatial"].get("per_class", {}).get(k, {"count": 0, "area_px": 0}); c.drawString(50, y, f"{LESION_NAMES[k]}: count {s['count']}, area {s['area_px']} px"); y -= 12
    ov = overlay(res["preprocessed"], res["masks"]); b = io.BytesIO(); Image.fromarray(ov).save(b, "PNG"); b.seek(0)
    c.drawImage(ImageReader(b), 40, 60, width=300, height=300)
    c.setFont("Helvetica-Oblique", 8); c.drawString(40, 40, "Limitations: dataset-bound performance; dataset shift; segmentation classes do not cover all DR signs; confidence != certainty.")
    c.showPage(); c.save(); return buf.getvalue()
