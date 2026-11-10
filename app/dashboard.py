"""RETINA-LENS AI dashboard.  streamlit run app/dashboard.py
Real inference needs a trained checkpoint. 'Demonstration mode' uses SYNTHETIC data and is always labelled."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retina_lens.data.preprocess import fundus_mask
from retina_lens.data.schema import GRADE_NAMES, LESION_NAMES, LESIONS
from retina_lens.explain import COLORS, overlay
from retina_lens.inference import CheckpointMissing, Predictor
from retina_lens.report import DISCLAIMER, build_pdf
from retina_lens.spatial import analyze, derive_boxes

st.set_page_config("RETINA-LENS AI", "👁️", layout="wide")
NAVY, TEAL = "#0b1630", "#2dd4bf"
st.markdown(f"""<style>
.block-container{{padding-top:1.5rem}} h1,h2,h3{{letter-spacing:-.01em}}
.card{{background:linear-gradient(145deg,#132447,#0e1d3b);border:1px solid #1f3560;border-radius:16px;padding:18px 20px;margin-bottom:12px}}
.kpi{{font-size:2rem;font-weight:700;color:{TEAL}}} .sub{{color:#9fb3d9;font-size:.85rem}}
.banner{{background:#3b1d2a;border:1px solid #e11d48;color:#ffd1dc;padding:8px 14px;border-radius:10px;font-size:.85rem;margin-bottom:10px}}
.ok{{background:#0d3b3a;border:1px solid {TEAL};color:#c9fff7}}
</style>""", unsafe_allow_html=True)

CKPT = os.getenv("RETINA_CHECKPOINT", "runs/convnext_coral_mt/best.pt")


@st.cache_resource
def get_predictor(path: str):
    try:
        return Predictor(path, os.getenv("RETINA_DEVICE", "auto")), None
    except CheckpointMissing as e:
        return None, str(e)


def kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(f"<div class='card'><div class='sub'>{label}</div><div class='kpi'>{value}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def demo_result() -> tuple[np.ndarray, dict]:
    """SYNTHETIC demonstration only - not a model output."""
    s = 384; rng = np.random.default_rng(7); yy, xx = np.mgrid[:s, :s]; disc = ((yy - s / 2) ** 2 + (xx - s / 2) ** 2) < (s / 2.05) ** 2
    img = np.zeros((s, s, 3), np.uint8); img[disc] = (150, 70, 30); masks = np.zeros((4, s, s), bool)
    for c, n in zip(range(4), (25, 10, 12, 4)):
        for _ in range(n):
            cy, cx = rng.integers(60, s - 60, 2); r = int(rng.integers(2, 7))
            masks[c] |= (((yy - cy) ** 2 + (xx - cx) ** 2) < r * r) & disc
    p = np.array([.05, .1, .6, .2, .05])
    return img, {"model_version": "DEMO (synthetic, not a model)", "grade": 2, "probabilities": p.tolist(), "entropy": float(-(p * np.log(p)).sum()),
                 "abstain": False, "quality": {"blur_var": 99.0, "brightness": 120.0, "fundus_fraction": .7, "overexposed_frac": 0, "heuristic_adequate": True, "learned_quality_prob": 0.9},
                 "preprocessed": img, "masks": masks, "boxes": derive_boxes(masks), "spatial": analyze(masks, fundus_mask(img)), "inference_ms": 0.0,
                 "cam": np.clip(1 - np.hypot(yy - s / 2, xx - s / 2) / (s / 2), 0, 1).astype(np.float32), "synthetic_model": True}


st.sidebar.markdown("## 👁️ RETINA-LENS AI")
page = st.sidebar.radio("Navigate", ["Overview", "Retinal Image Analysis", "DR Severity Grading", "Lesion Explorer", "Spatial Analysis",
                                      "Explainability", "Model Performance", "Experiment Comparison", "Model Information"])
demo = st.sidebar.toggle("Demonstration mode (synthetic)", False, help="Shows fake data clearly labelled as such.")
predictor, err = get_predictor(CKPT)
st.sidebar.markdown(f"<div class='banner {'ok' if predictor else ''}'>{'Model loaded' if predictor else 'Inference unavailable'}</div>", unsafe_allow_html=True)
st.sidebar.caption(DISCLAIMER)

if demo:
    st.markdown("<div class='banner'>⚠️ DEMONSTRATION MODE: synthetic image and invented outputs. NOT produced by a trained model.</div>", unsafe_allow_html=True)


def current_result():
    if demo:
        return demo_result()
    if predictor is None:
        st.error(f"Inference unavailable: {err}. Train a model (see README) or enable Demonstration mode."); return None, None
    f = st.file_uploader("Upload fundus photograph (JPEG/PNG)", type=["jpg", "jpeg", "png"], key="up")
    if f is None:
        st.info("Upload an image to begin."); return None, None
    arr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        st.error("Could not decode image."); return None, None
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    key = (f.name, rgb.shape)
    if st.session_state.get("key") != key:
        with st.spinner("Running multi-task inference..."):
            st.session_state["res"] = predictor.predict(rgb, cam=True); st.session_state["key"] = key; st.session_state["orig"] = rgb
    return st.session_state["orig"], st.session_state["res"]


if page == "Overview":
    st.title("RETINA-LENS AI")
    st.caption("Multi-task retinal analysis: DR grading · lesion segmentation · quality · uncertainty · explainability")
    c = st.columns(4)
    with c[0]: kpi("Tasks", "5", "grading, segmentation, detection*, quality, uncertainty")
    with c[1]: kpi("Lesion classes", "4", "MA · HE · EX · SE")
    with c[2]: kpi("Severity grades", "5", "ordinal (CORAL)")
    with c[3]: kpi("Status", "Ready" if predictor else "No model", CKPT)
    st.markdown("""<div class='card'><b>How it works.</b> A shared encoder (ConvNeXt / EfficientNetV2 / Swin) feeds an FPN decoder with a high-resolution pathway for lesion
    segmentation, and a grading head that cross-attends to lesion tokens while retaining global context. Missing annotations are masked from the loss, never treated as negatives.
    <br><br><span class='sub'>*Boxes are derived from segmentation components; a standalone detector (Faster R-CNN/RetinaNet) is a documented extension.
    Research prototype - not a medical device, not clinically validated.</span></div>""", unsafe_allow_html=True)

elif page == "Retinal Image Analysis":
    st.title("Retinal Image Analysis")
    orig, r = current_result()
    if r:
        a, b = st.columns(2); a.image(orig if orig is not None else r["preprocessed"], caption="Original", use_container_width=True)
        b.image(r["preprocessed"], caption="Preprocessed (crop · pad · CLAHE)", use_container_width=True)
        k = st.columns(4)
        with k[0]: kpi("Predicted severity", f"{r['grade']} · {GRADE_NAMES[r['grade']]}")
        with k[1]: kpi("Entropy", f"{r['entropy']:.2f}", "higher = less certain")
        with k[2]: kpi("Quality", "Adequate" if r["quality"]["heuristic_adequate"] else "Inadequate", f"learned p={r['quality']['learned_quality_prob']:.2f}")
        with k[3]: kpi("Latency", f"{r['inference_ms']:.0f} ms")
        if r["abstain"]:
            st.warning("Abstain flag: uncertainty is high or image quality is inadequate. Do not rely on this output.")
        show = {k_: st.sidebar.checkbox(LESION_NAMES[k_], True, key=f"s_{k_}") for k_ in LESIONS}
        st.image(overlay(r["preprocessed"], r["masks"], show, boxes=r["boxes"]), caption="Lesion overlay + derived boxes", use_container_width=True)
        st.download_button("Download PDF report", build_pdf("uploaded" if not demo else "DEMO", r), "retina_lens_report.pdf", "application/pdf")

elif page == "DR Severity Grading":
    st.title("DR Severity Grading")
    _, r = current_result()
    if r:
        fig = go.Figure(go.Bar(x=GRADE_NAMES, y=r["probabilities"], marker_color=[TEAL if i == r["grade"] else "#2b4377" for i in range(5)]))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1], title="Class probabilities")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Probabilities express model confidence, not clinical certainty.")

elif page == "Lesion Explorer":
    st.title("Lesion Explorer")
    _, r = current_result()
    if r:
        sel = st.multiselect("Lesion classes", LESIONS, LESIONS, format_func=lambda k: LESION_NAMES[k])
        st.image(overlay(r["preprocessed"], r["masks"], {k: k in sel for k in LESIONS}, boxes=r["boxes"]), use_container_width=True)
        st.dataframe(pd.DataFrame([{"Lesion": LESION_NAMES[k], **v} for k, v in r["spatial"]["per_class"].items()]), use_container_width=True)
        st.dataframe(pd.DataFrame(r["boxes"]), use_container_width=True, height=240)
        st.caption("Boxes are derived from connected components of the predicted masks.")

elif page == "Spatial Analysis":
    st.title("Spatial Analysis")
    _, r = current_result()
    if r:
        L = pd.DataFrame(r["spatial"]["lesions"]); fig = go.Figure()
        s = r["preprocessed"].shape[0]; fig.add_shape(type="circle", x0=0, y0=0, x1=s, y1=s, line_color="#2b4377")
        for rr in (.33, .66):
            c = r["spatial"]["reference"]["center"]; R = r["spatial"]["reference"]["radius_px"] * rr
            fig.add_shape(type="circle", x0=c[0] - R, y0=c[1] - R, x1=c[0] + R, y1=c[1] + R, line=dict(color="#1f3560", dash="dot"))
        if len(L):
            for k in LESIONS:
                d = L[L.lesion == k]
                fig.add_trace(go.Scattergl(x=d.x, y=d.y, mode="markers", name=LESION_NAMES[k], marker=dict(size=7, color="rgb%s" % (COLORS[k],)),
                                           customdata=d[["zone", "area_px"]], hovertemplate="%{customdata[0]} zone<br>area %{customdata[1]} px<extra></extra>"))
        fig.update_yaxes(autorange="reversed", scaleanchor="x"); fig.update_layout(template="plotly_dark", height=560, paper_bgcolor="rgba(0,0,0,0)", title="Interactive lesion map (image coordinates)")
        st.plotly_chart(fig, use_container_width=True)
        if len(L):
            st.dataframe(L.groupby(["lesion", "zone"]).size().unstack(fill_value=0), use_container_width=True)
        k = st.selectbox("Density map", LESIONS, format_func=lambda k: LESION_NAMES[k])
        st.plotly_chart(go.Figure(go.Heatmap(z=r["spatial"]["density"][k], colorscale="Teal")).update_yaxes(autorange="reversed").update_layout(template="plotly_dark", height=420), use_container_width=True)
        st.caption("Normalised pixel units only; no calibrated anatomical measurements.")

elif page == "Explainability":
    st.title("Explainability")
    orig, r = current_result()
    if r and "cam" in r:
        a = st.slider("Heatmap blend", 0.0, 1.0, 0.5); base = r["preprocessed"]
        hm = cv2.cvtColor(cv2.applyColorMap((r["cam"] * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        c = st.columns(2); c[0].image(cv2.addWeighted(base, 1 - a, hm, a, 0), caption="Grad-CAM (attribution)", use_container_width=True)
        c[1].image(overlay(base, r["masks"]), caption="Predicted lesion segmentation", use_container_width=True)
        st.info("Grad-CAM is an attribution heatmap of the grading output. It is NOT lesion localisation and does not prove the model's reasoning. "
                "Segmentation masks are predictions trained on lesion annotations; only expert ground truth is ground truth.")

elif page == "Model Performance":
    st.title("Model Performance")
    files = sorted(Path("runs").glob("*/metrics_test.json"))
    if not files:
        st.warning("No evaluation results found. Run `python -m retina_lens.evaluate --run runs/<name>`. Nothing is shown rather than inventing numbers.")
    else:
        f = st.selectbox("Run", files, format_func=lambda p: p.parent.name); m = json.loads(f.read_text())
        g = m["grading"]; c = st.columns(4)
        with c[0]: kpi("QWK", f"{g['qwk']:.3f}", f"95% CI {m['qwk_ci95'][0]:.3f}-{m['qwk_ci95'][1]:.3f}")
        with c[1]: kpi("Macro F1", f"{g['macro_f1']:.3f}")
        with c[2]: kpi("ECE (T-scaled)", f"{m['calibration_temp_scaled']['ece']:.3f}", f"raw {m['calibration_raw']['ece']:.3f}")
        with c[3]: kpi("n", str(m["n"]), f"split: {m['split']}")
        st.plotly_chart(go.Figure(go.Heatmap(z=g["confusion_matrix"], x=GRADE_NAMES, y=GRADE_NAMES, colorscale="Teal", text=g["confusion_matrix"], texttemplate="%{text}")).update_layout(template="plotly_dark", title="Confusion matrix (rows=true)", yaxis_autorange="reversed"), use_container_width=True)
        rel = m["calibration_temp_scaled"]["reliability"]
        if rel:
            fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="perfect"), go.Scatter(x=[r[0] for r in rel], y=[r[1] for r in rel], mode="lines+markers", name="model")])
            st.plotly_chart(fig.update_layout(template="plotly_dark", title="Reliability diagram", xaxis_title="confidence", yaxis_title="accuracy"), use_container_width=True)

elif page == "Experiment Comparison":
    st.title("Experiment Comparison")
    p = Path("results/comparison.csv")
    if not p.exists():
        st.warning("Run `python scripts/compare.py` after training experiments.")
    else:
        df = pd.read_csv(p); st.dataframe(df, use_container_width=True)
        y = "test_qwk" if "test_qwk" in df and df.test_qwk.notna().any() else "val_qwk"
        st.plotly_chart(go.Figure(go.Bar(x=df.run, y=df[y], marker_color=TEAL)).update_layout(template="plotly_dark", title=y), use_container_width=True)

elif page == "Model Information":
    st.title("Model Information")
    if predictor:
        st.json({"version": predictor.version, "backbone": predictor.cfg["model"]["backbone"], "grade_head": predictor.mode, "img_size": predictor.size,
                 "temperature": predictor.temperature, "trained_on_synthetic_smoke_test": predictor.synthetic})
    else:
        st.warning(f"No checkpoint: {err}")
    st.markdown("**Intended use:** research and education. **Not** for diagnosis or patient care. See MODEL_CARD.md.")
