from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.core.config import settings
from app.services.image_model import VisionBackend, create_vision_backend

logger = logging.getLogger(__name__)

_MAX_SIDE = 512
_MAX_BYTES = 12 * 1024 * 1024


class ImageService:
    """Decode, preprocess, and extract lightweight visual claim signals."""

    def __init__(self, *, backend: VisionBackend | None = None) -> None:
        self._backend = backend or create_vision_backend(settings.image_model_type)

    def _decode_rgb(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("empty image bytes")
        if len(image_bytes) > _MAX_BYTES:
            raise ValueError(f"image exceeds max size ({_MAX_BYTES} bytes)")
        bio = io.BytesIO(image_bytes)
        with Image.open(bio) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("image must decode to RGB")
        return arr

    def analyze(self, image_bytes: bytes) -> dict[str, Any]:
        """Return damage_type, severity, confidence, plus diagnostics."""
        err: dict[str, Any] = {
            "damage_type": "unknown",
            "severity": "low",
            "confidence": 0.0,
            "backend": "error",
            "signals": {},
        }
        try:
            rgb = self._decode_rgb(image_bytes)
            out = self._backend.analyze_rgb(rgb)
        except Exception as exc:
            logger.warning(
                "image_analyze_failed",
                extra={"error": type(exc).__name__},
            )
            return err
        if not isinstance(out, dict):
            return err
        damage_type = str(out.get("damage_type") or "unknown")
        severity = str(out.get("severity") or "low").lower()
        if severity not in ("low", "medium", "high"):
            severity = "medium"
        try:
            conf = float(out.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(1.0, max(0.0, conf))
        return {
            "damage_type": damage_type,
            "severity": severity,
            "confidence": round(conf, 4),
            "backend": str(out.get("backend") or "unknown"),
            "signals": out.get("signals") if isinstance(out.get("signals"), dict) else {},
        }


__all__ = ["ImageService"]
