"""Shared encoder + FPN decoder + high-res pathway + grading/quality heads with lesion-aware cross-task attention."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(i: int, o: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class FPNDecoder(nn.Module):
    """Top-down FPN with multi-scale fusion at the highest-resolution encoder stride."""

    def __init__(self, chs: list[int], dim: int):
        super().__init__()
        self.lat = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in chs)
        self.smooth = nn.ModuleList(conv_bn(dim, dim) for _ in chs)
        self.fuse = conv_bn(dim * len(chs), dim)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        p = [l(f) for l, f in zip(self.lat, feats)]
        for i in range(len(p) - 1, 0, -1):  # skip connections via lateral adds
            p[i - 1] = p[i - 1] + F.interpolate(p[i], size=p[i - 1].shape[-2:], mode="nearest")
        p = [s(x) for s, x in zip(self.smooth, p)]
        t = p[0].shape[-2:]
        return self.fuse(torch.cat([p[0]] + [F.interpolate(x, size=t, mode="bilinear", align_corners=False) for x in p[1:]], 1))


class CoralHead(nn.Module):
    """CORAL (Cao et al. 2020): one shared weight vector + K-1 ordered biases -> rank-consistent logits."""

    def __init__(self, in_dim: int, k: int = 5):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False); self.bias = nn.Parameter(torch.zeros(k - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.bias


class MultiTaskNet(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True, img_size: int = 512,
                 decoder_dim: int = 128, dropout: float = 0.2, lesion_attention: bool = True,
                 grade_mode: str = "coral", n_lesions: int = 4, n_grades: int = 5):
        super().__init__()
        kw = {"img_size": img_size} if "swin" in backbone else {}
        idx = (1, 2, 3, 4) if "efficientnet" in backbone else (0, 1, 2, 3)
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=idx, **kw)
        chs = self.encoder.feature_info.channels()
        D = decoder_dim
        self.decoder = FPNDecoder(chs, D)
        self.hires = nn.Sequential(conv_bn(3, 32, 3, 2), conv_bn(32, 32))          # stride-2 detail path for MAs
        self.seg_refine = nn.Sequential(conv_bn(D + 32, D), nn.Conv2d(D, n_lesions, 1))
        self.aux_seg = nn.Conv2d(D, n_lesions, 1)                                    # deep supervision (stride 4)
        self.proj = nn.Linear(chs[-1], D)
        self.use_attn, self.grade_mode, self.n_lesions = lesion_attention, grade_mode, n_lesions
        if lesion_attention:
            self.attn = nn.MultiheadAttention(D, 4, batch_first=True)
        fd = D * 2 if lesion_attention else D
        self.drop = nn.Dropout(dropout)  # also used for MC-dropout
        if grade_mode == "coral":
            self.grade_head = CoralHead(fd, n_grades)
        elif grade_mode == "cumulative":
            self.grade_head = nn.Linear(fd, n_grades - 1)
        else:
            self.grade_head = nn.Linear(fd, n_grades)
        self.quality_head = nn.Linear(D, 1)
        self.last_feat: torch.Tensor | None = None  # for Grad-CAM

    @staticmethod
    def _nchw(f: torch.Tensor, c: int) -> torch.Tensor:
        return f.permute(0, 3, 1, 2).contiguous() if f.shape[1] != c and f.shape[-1] == c else f

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        chs = self.encoder.feature_info.channels()
        feats = [self._nchw(f, c) for f, c in zip(self.encoder(x), chs)]
        self.last_feat = feats[-1]
        dec = self.decoder(feats)                                                   # B,D,H/4,W/4
        up = F.interpolate(dec, scale_factor=2, mode="bilinear", align_corners=False)
        hr = self.hires(x)
        if up.shape[-2:] != hr.shape[-2:]:
            up = F.interpolate(up, size=hr.shape[-2:], mode="bilinear", align_corners=False)
        seg_half = self.seg_refine(torch.cat([up, hr], 1))
        seg = F.interpolate(seg_half, size=x.shape[-2:], mode="bilinear", align_corners=False)
        aux = F.interpolate(self.aux_seg(dec), size=x.shape[-2:], mode="bilinear", align_corners=False)
        g = self.proj(feats[-1].mean((2, 3)))                                       # global retinal context
        if self.use_attn:  # lesion tokens = segmentation-probability-weighted pooling of decoder features
            pr = F.interpolate(torch.sigmoid(seg_half), size=dec.shape[-2:], mode="bilinear", align_corners=False)
            tok = (pr.unsqueeze(2) * dec.unsqueeze(1)).sum((3, 4)) / (pr.sum((2, 3)).unsqueeze(-1) + 1e-6)
            ctx, _ = self.attn(g.unsqueeze(1), tok, tok)
            gf = torch.cat([g, ctx[:, 0]], 1)                                        # keeps global info + lesion info
        else:
            gf = g
        return {"grade_logits": self.grade_head(self.drop(gf)), "seg_logits": seg, "aux_seg_logits": aux,
                "quality_logit": self.quality_head(g).squeeze(-1)}


def grade_probs(logits: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert head output to 5-class probabilities for every grading mode."""
    if mode in ("coral", "cumulative"):
        c = torch.sigmoid(logits)                                                    # P(y > k)
        c = torch.cummin(c, dim=1).values                                            # enforce monotonic
        one, zero = torch.ones_like(c[:, :1]), torch.zeros_like(c[:, :1])
        p = torch.cat([one, c], 1) - torch.cat([c, zero], 1)
        return p.clamp_min(1e-8) / p.clamp_min(1e-8).sum(1, keepdim=True)
    return F.softmax(logits, 1)


def build_model(cfg: dict) -> MultiTaskNet:
    m, l = cfg["model"], cfg["loss"]
    mode = l["grade"] if l["grade"] in ("coral", "cumulative") else "ce"
    return MultiTaskNet(m["backbone"], m["pretrained"], cfg["data"]["img_size"], m["decoder_dim"], m["dropout"],
                        m["lesion_attention"], mode)
