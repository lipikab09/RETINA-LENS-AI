import numpy as np
import pandas as pd
import pytest
import torch

from retina_lens.data.preprocess import fundus_bbox, quality_metrics, resize_pad
from retina_lens.data.schema import COLUMNS, make_record
from retina_lens.data.splits import assign_splits
from retina_lens.losses import grade_loss, seg_loss
from retina_lens.metrics import calibration, qwk
from retina_lens.models.multitask import MultiTaskNet, grade_probs
from retina_lens.spatial import analyze, derive_boxes
from retina_lens.uncertainty import apply_temperature


def test_resize_pad_keeps_aspect():
    out = resize_pad(np.ones((100, 200, 3), np.uint8), 64); assert out.shape == (64, 64, 3) and out[0].sum() == 0


def test_bbox_and_quality():
    img = np.zeros((100, 100, 3), np.uint8); img[20:80, 30:90] = 120
    assert fundus_bbox(img) == (20, 80, 30, 90) and quality_metrics(img)["fundus_fraction"] > 0.3


def test_missing_labels_are_not_negative():
    r = make_record("x", "a", "p.png", 10, 10)
    assert not any(r[f"valid_{k}"] for k in ["MA", "HE", "EX", "SE"]) and r["grade_valid"] is False and set(r) == set(COLUMNS)


def test_masked_losses_ignore_invalid():
    lg = torch.randn(2, 4, 8, 8, requires_grad=True); tgt = torch.ones(2, 4, 8, 8)
    assert seg_loss(lg, tgt, torch.zeros(2, 4, dtype=torch.bool)).item() == 0.0
    gl = torch.randn(2, 4, requires_grad=True)
    assert grade_loss(gl, torch.tensor([1, 2]), torch.zeros(2, dtype=torch.bool), "coral").item() == 0.0


def test_ordinal_probs_sum_to_one():
    p = grade_probs(torch.randn(6, 4), "coral"); assert torch.allclose(p.sum(1), torch.ones(6), atol=1e-5) and (p >= 0).all()


@pytest.mark.parametrize("attn", [True, False])
def test_model_forward_cpu(attn):
    m = MultiTaskNet("convnext_tiny", False, 64, 16, 0.1, attn, "coral").eval()
    o = m(torch.randn(2, 3, 64, 64))
    assert o["grade_logits"].shape == (2, 4) and o["seg_logits"].shape == (2, 4, 64, 64) and o["quality_logit"].shape == (2,)


def test_group_split_has_no_patient_leakage():
    df = pd.DataFrame({"dataset": "d", "image_id": [str(i) for i in range(200)], "patient_id": [str(i // 2) for i in range(200)],
                       "grade_valid": True, "grade": np.arange(200) % 5})
    s = assign_splits(df, 0); assert (s.groupby("patient_id").split.nunique() == 1).all()


def test_spatial_and_boxes():
    m = np.zeros((4, 64, 64), bool); m[0, 10:13, 10:13] = True; f = np.ones((64, 64), np.uint8)
    assert derive_boxes(m)[0]["lesion"] == "MA" and analyze(m, f)["per_class"]["MA"]["count"] == 1


def test_metrics_and_temperature():
    y = np.array([0, 1, 2, 3, 4]); p = np.eye(5)[y] * 0.9 + 0.025
    assert qwk(y, p.argmax(1)) == 1.0 and calibration(y, p)["ece"] < 0.2
    assert np.allclose(apply_temperature(p, 1.0).sum(1), 1.0)
