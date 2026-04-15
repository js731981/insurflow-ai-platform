from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VisionBackend(ABC):
    """Pluggable vision analysis (heuristic CV or optional CNN features)."""

    @abstractmethod
    def analyze_rgb(self, rgb: np.ndarray) -> dict[str, Any]:
        """``rgb`` is uint8 H×W×3 RGB."""


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel-like magnitude via ``np.gradient`` (no OpenCV)."""
    g = np.gradient(gray.astype(np.float32))
    gx, gy = g[1], g[0]
    return np.hypot(gx, gy)


def _heuristic_from_visuals(gray: np.ndarray, edge_mag: np.ndarray) -> dict[str, Any]:
    h, w = gray.shape[:2]
    pixels = float(h * w) if h and w else 1.0

    mean_l = float(np.mean(gray)) / 255.0
    edge_mean = float(np.mean(edge_mag)) / 255.0
    edge_p95 = float(np.percentile(edge_mag, 95)) / 255.0

    # Thin high-gradient structures (proxy for cracks / shattered glass).
    strong = edge_mag > (0.35 * 255.0)
    strong_frac = float(np.mean(strong.astype(np.float32)))

    crack_score = _clamp01(1.6 * edge_p95 + 0.9 * strong_frac - 0.15 * mean_l)

    if crack_score >= 0.42:
        damage_type = "screen_crack"
    elif edge_mean >= 0.12 and strong_frac >= 0.08:
        damage_type = "impact_marks"
    elif edge_mean >= 0.07:
        damage_type = "surface_wear"
    else:
        damage_type = "unknown"

    severity_score = _clamp01(0.55 * crack_score + 0.35 * edge_mean + 0.25 * strong_frac)
    if severity_score < 0.28:
        severity = "low"
    elif severity_score < 0.55:
        severity = "medium"
    else:
        severity = "high"

    confidence = _clamp01(0.35 + 0.55 * max(crack_score, edge_mean))

    return {
        "damage_type": damage_type,
        "severity": severity,
        "confidence": round(confidence, 4),
        "backend": "heuristic",
        "signals": {
            "edge_density": round(edge_mean, 4),
            "edge_p95": round(edge_p95, 4),
            "strong_edge_fraction": round(strong_frac, 4),
            "brightness_norm": round(mean_l, 4),
            "crack_score": round(crack_score, 4),
        },
    }


class HeuristicVisionBackend(VisionBackend):
    def analyze_rgb(self, rgb: np.ndarray) -> dict[str, Any]:
        gray = _rgb_to_gray(rgb)
        mag = _edge_magnitude(gray)
        out = _heuristic_from_visuals(gray, mag)
        return out


class CnnFeatureVisionBackend(VisionBackend):
    """Uses MobileNetV2 pooled activations as a complexity proxy (no domain training).

    Falls back to :class:`HeuristicVisionBackend` if torch/torchvision is unavailable.
    """

    _model = None
    _tfm = None

    def __init__(self) -> None:
        self._fallback = HeuristicVisionBackend()

    def _lazy_load(self) -> tuple[Any, Any] | None:
        if CnnFeatureVisionBackend._model is not None:
            return CnnFeatureVisionBackend._model, CnnFeatureVisionBackend._tfm
        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms
        except Exception as exc:  # pragma: no cover - optional stack
            logger.info("cnn_vision_backend_unavailable", extra={"error": f"{type(exc).__name__}: {exc}"})
            return None

        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            net = models.mobilenet_v2(weights=weights)
        except Exception as exc:  # pragma: no cover
            logger.info("cnn_mobilenet_load_failed", extra={"error": f"{type(exc).__name__}: {exc}"})
            return None

        net.classifier = torch.nn.Identity()  # type: ignore[assignment]
        net.eval()
        tfm = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        CnnFeatureVisionBackend._model = net
        CnnFeatureVisionBackend._tfm = tfm
        return net, tfm

    def analyze_rgb(self, rgb: np.ndarray) -> dict[str, Any]:
        loaded = self._lazy_load()
        if loaded is None:
            h = self._fallback.analyze_rgb(rgb)
            h["backend"] = "heuristic_cnn_fallback"
            return h

        net, tfm = loaded
        try:
            import torch
        except Exception:  # pragma: no cover
            return self._fallback.analyze_rgb(rgb)

        # HWC uint8 RGB
        t = tfm(rgb).unsqueeze(0)
        with torch.no_grad():
            feats = net(t).squeeze(0).float()
        z = float(torch.mean(torch.abs(feats)).item())
        # Map feature activity to proxy scores (deterministic, bounded).
        edge_proxy = _clamp01(math.tanh(z / 2.8))
        complexity = _clamp01(math.tanh(z / 2.0))

        gray = _rgb_to_gray(rgb)
        mag = _edge_magnitude(gray)
        heur = _heuristic_from_visuals(gray, mag)

        fused_crack = _clamp01(0.55 * float(heur["signals"]["crack_score"]) + 0.45 * complexity)
        if fused_crack >= 0.45:
            damage_type = "screen_crack"
        elif edge_proxy >= 0.35:
            damage_type = "impact_marks"
        else:
            damage_type = str(heur.get("damage_type") or "unknown")

        sev_h = str(heur.get("severity") or "medium")
        if complexity >= 0.62:
            severity = "high"
        elif complexity >= 0.35:
            severity = "medium" if sev_h == "low" else sev_h
        else:
            severity = sev_h

        confidence = _clamp01(0.4 + 0.45 * max(complexity, float(heur.get("confidence") or 0.0)))

        out = {
            "damage_type": damage_type,
            "severity": severity,
            "confidence": round(confidence, 4),
            "backend": "cnn_mobilenet_v2_features",
            "signals": {
                **(heur.get("signals") or {}),
                "cnn_feature_activity": round(z, 4),
                "cnn_complexity": round(complexity, 4),
            },
        }
        return out


def create_vision_backend(model_type: str) -> VisionBackend:
    mt = (model_type or "heuristic").strip().lower()
    if mt == "cnn":
        return CnnFeatureVisionBackend()
    return HeuristicVisionBackend()


__all__ = [
    "VisionBackend",
    "HeuristicVisionBackend",
    "CnnFeatureVisionBackend",
    "create_vision_backend",
]
