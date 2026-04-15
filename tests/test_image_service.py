"""Tests for ImageService decoding and analysis."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.services.image_model import HeuristicVisionBackend, VisionBackend
from app.services.image_service import ImageService


def create_blank_png() -> bytes:
    img = Image.new("RGB", (64, 64), color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_noise_png() -> bytes:
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _assert_analyze_shape(result: dict) -> None:
    assert "damage_type" in result
    assert "severity" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_analyze_blank_image_heuristic():
    svc = ImageService(backend=HeuristicVisionBackend())
    result = svc.analyze(create_blank_png())
    _assert_analyze_shape(result)
    assert result["severity"] in ("low", "medium", "high")


def test_analyze_noise_image_heuristic():
    svc = ImageService(backend=HeuristicVisionBackend())
    result = svc.analyze(create_noise_png())
    _assert_analyze_shape(result)


def test_analyze_corrupt_bytes_returns_safe_default():
    svc = ImageService(backend=HeuristicVisionBackend())
    result = svc.analyze(b"\x00\x01not-a-valid-image\xff")
    assert result == {
        "damage_type": "unknown",
        "severity": "low",
        "confidence": 0.0,
        "backend": "error",
        "signals": {},
    }


class _BadBackend(VisionBackend):
    def analyze_rgb(self, rgb: np.ndarray) -> dict:
        raise RuntimeError("simulated backend failure")


def test_analyze_backend_exception_returns_safe_default():
    svc = ImageService(backend=_BadBackend())
    result = svc.analyze(create_blank_png())
    assert result["backend"] == "error"
    assert result["confidence"] == 0.0
    assert result["damage_type"] == "unknown"


def test_analyze_non_dict_backend_output_normalized():
    class _WeirdBackend(VisionBackend):
        def analyze_rgb(self, rgb: np.ndarray):
            return None  # type: ignore[return-value]

    svc = ImageService(backend=_WeirdBackend())
    result = svc.analyze(create_blank_png())
    assert result["backend"] == "error"
    _assert_analyze_shape(result)
