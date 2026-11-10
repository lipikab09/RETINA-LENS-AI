"""FastAPI service.  uvicorn retina_lens.api.main:app --port 8000"""
from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from ..data.schema import GRADE_NAMES
from ..explain import overlay
from ..inference import CheckpointMissing, Predictor
from ..utils import get_logger

log = get_logger("api")
STATE: dict = {"predictor": None, "error": None}
MAX_MB = int(os.getenv("RETINA_MAX_UPLOAD_MB", "15"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        STATE["predictor"] = Predictor(os.getenv("RETINA_CHECKPOINT", "runs/convnext_coral_mt/best.pt"), os.getenv("RETINA_DEVICE", "auto"))
    except CheckpointMissing as e:
        STATE["error"] = str(e); log.warning("inference unavailable: %s", e)
    yield


app = FastAPI(title="RETINA-LENS AI", version="0.1.0", lifespan=lifespan,
              description="Research prototype. Outputs are research predictions, not medical diagnoses.")


class Box(BaseModel):
    lesion: str; x0: int; y0: int; x1: int; y1: int; area_px: int; confidence: float


class PredictResponse(BaseModel):
    model_version: str; grade: int; grade_name: str; probabilities: list[float]; entropy: float; abstain: bool
    image_quality_adequate: bool; lesion_counts: dict[str, int]; boxes: list[Box]; inference_ms: float
    overlay_png_b64: str | None = None; disclaimer: str = "Research prediction, not a medical diagnosis."


@app.get("/health")
def health():
    return {"status": "ok" if STATE["predictor"] else "degraded", "inference_available": STATE["predictor"] is not None, "detail": STATE["error"]}


@app.get("/model-info")
def model_info():
    p = STATE["predictor"]
    if p is None:
        raise HTTPException(503, STATE["error"] or "no model")
    return {"version": p.version, "backbone": p.cfg["model"]["backbone"], "grade_head": p.mode, "img_size": p.size,
            "temperature": p.temperature, "device": str(p.dev), "trained_on_synthetic_smoke_test": p.synthetic}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), overlay_png: bool = True, mc: bool = True):
    p = STATE["predictor"]
    if p is None:
        raise HTTPException(503, f"Inference unavailable: {STATE['error']}")
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Only JPEG/PNG images are accepted")
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_MB} MB")
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None or min(arr.shape[:2]) < 128:
        raise HTTPException(422, "Undecodable or too-small image")
    r = p.predict(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), mc=mc)
    b64 = None
    if overlay_png:
        buf = io.BytesIO(); Image.fromarray(overlay(r["preprocessed"], r["masks"])).save(buf, "PNG"); b64 = base64.b64encode(buf.getvalue()).decode()
    log.info("predict %s grade=%d ms=%.0f", file.filename, r["grade"], r["inference_ms"])
    return PredictResponse(model_version=r["model_version"], grade=r["grade"], grade_name=GRADE_NAMES[r["grade"]], probabilities=r["probabilities"],
                           entropy=r["entropy"], abstain=r["abstain"], image_quality_adequate=r["quality"]["heuristic_adequate"],
                           lesion_counts={k: v["count"] for k, v in r["spatial"]["per_class"].items()}, boxes=r["boxes"],
                           inference_ms=r["inference_ms"], overlay_png_b64=b64)


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    log.exception("unhandled"); return JSONResponse({"error": "internal_error", "detail": str(exc)}, 500)
