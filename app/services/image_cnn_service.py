from __future__ import annotations

import io
import logging
import time
from typing import Any, Literal

from PIL import Image, ImageOps

from app.services.image_service import ImageService
from app.core.config import CNN_CONF_THRESHOLD

logger = logging.getLogger(__name__)

Label = Literal["no_damage", "minor_crack", "major_crack"]
Severity = Literal["low", "medium", "high"]


class ImageCNNService:
    """CNN-based mobile screen damage classification with safe heuristic fallback.

    Notes:
    - Uses MobileNetV2 backbone for CPU-friendly inference.
    - Expects (optional) fine-tuned weights for the 3-class head. If unavailable,
      falls back to the existing heuristic `ImageService` to avoid breaking the pipeline.
    """

    _model: Any | None = None
    _transform: Any | None = None
    _idx_to_class: list[str] | None = None
    _load_error_logged: bool = False

    def __init__(
        self,
        *,
        model_path: str | None = None,
        device: str | None = None,
        fallback_service: ImageService | None = None,
    ) -> None:
        # Optional path to a fine-tuned state_dict (recommended).
        # If not provided, the service will try env var IMAGE_CNN_MODEL_PATH.
        self._model_path = model_path
        self._device = device  # e.g. "cpu"
        self._fallback = fallback_service or ImageService()

    @staticmethod
    def _severity_from_label(label: str) -> Severity:
        m = {
            "no_damage": "low",
            "minor_crack": "medium",
            "major_crack": "high",
        }
        return m.get(str(label).strip().lower(), "medium")  # type: ignore[return-value]

    def _lazy_load(self) -> tuple[Any, Any, Any] | None:
        if (
            ImageCNNService._model is not None
            and ImageCNNService._transform is not None
            and ImageCNNService._idx_to_class is not None
        ):
            try:
                import torch

                dev = next(ImageCNNService._model.parameters()).device  # type: ignore[union-attr]
            except Exception:
                dev = None
            return ImageCNNService._model, ImageCNNService._transform, dev

        try:
            import os

            import torch
            import torchvision.models as models
            from torchvision import transforms
        except Exception as exc:  # pragma: no cover - optional stack
            if not ImageCNNService._load_error_logged:
                ImageCNNService._load_error_logged = True
                logger.info(
                    "image_cnn_unavailable",
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                )
            return None

        # Device selection: keep CPU by default for predictability.
        dev_str = (self._device or "cpu").strip().lower()
        device = torch.device("cpu" if dev_str != "cuda" else "cuda")

        # Build model: ImageNet pretrained backbone, replace classifier for 3 classes.
        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            net = models.mobilenet_v2(weights=weights)
        except Exception as exc:  # pragma: no cover
            logger.info("image_cnn_backbone_load_failed", extra={"error": f"{type(exc).__name__}: {exc}"})
            return None

        net.classifier[1] = torch.nn.Linear(net.last_channel, 3)  # type: ignore[index]

        # Optional fine-tuned weights (state_dict). If missing, treat as unavailable and fall back.
        model_path = (
            self._model_path
            or os.getenv("IMAGE_CNN_MODEL_PATH", "").strip()
            or os.getenv("IMAGE_CNN_MODEL", "").strip()
            or "model.pth"
        )
        model_path = model_path.strip() or None
        if not model_path:
            logger.info("image_cnn_no_weights_configured")
            return None

        try:
            ckpt = torch.load(model_path, map_location="cpu")
            state: Any = ckpt
            idx_to_class: list[str] | None = None
            if isinstance(ckpt, dict):
                if isinstance(ckpt.get("idx_to_class"), list) and all(
                    isinstance(x, str) for x in ckpt["idx_to_class"]
                ):
                    idx_to_class = list(ckpt["idx_to_class"])
                if isinstance(ckpt.get("state_dict"), dict):
                    state = ckpt["state_dict"]
            net.load_state_dict(state, strict=False)
        except Exception as exc:
            logger.warning(
                "image_cnn_weights_load_failed",
                extra={"error": f"{type(exc).__name__}: {exc}", "model_path": model_path},
            )
            return None

        net.to(device)
        net.eval()

        tfm = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        ImageCNNService._model = net
        ImageCNNService._transform = tfm
        ImageCNNService._idx_to_class = idx_to_class or ["no_damage", "minor_crack", "major_crack"]
        return net, tfm, device

    @staticmethod
    def _decode_pil_rgb(image_bytes: bytes) -> Image.Image:
        if not image_bytes:
            raise ValueError("empty image bytes")
        bio = io.BytesIO(image_bytes)
        with Image.open(bio) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            return im.copy()

    def analyze(self, image_bytes: bytes) -> dict[str, Any]:
        """Classify damage severity from a phone screen image.

        Returns:
            {
              "label": "minor_crack",
              "confidence": 0.82,
              "severity": "low|medium|high",
              ...compat fields...
            }
        """
        t0 = time.perf_counter()
        loaded = self._lazy_load()
        if loaded is None:
            out = self._fallback.analyze(image_bytes)
            # Normalize to required output contract.
            sev = str(out.get("severity") or "low").strip().lower()
            if sev not in ("low", "medium", "high"):
                sev = "medium"
            return {
                "label": "unknown",
                "confidence": float(out.get("confidence") or 0.0),
                "severity": sev,
                "damage_type": str(out.get("damage_type") or "unknown"),
                "backend": "heuristic_fallback",
                "signals": {
                    **(out.get("signals") if isinstance(out.get("signals"), dict) else {}),
                    "fallback_reason": "cnn_unavailable",
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                },
            }

        net, tfm, device = loaded
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            logger.info("image_cnn_torch_missing", extra={"error": f"{type(exc).__name__}: {exc}"})
            return self._fallback.analyze(image_bytes)

        try:
            im = self._decode_pil_rgb(image_bytes)
            x = tfm(im).unsqueeze(0)
            if device is not None:
                x = x.to(device)
            with torch.no_grad():
                logits = net(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                conf_v, idx_v = torch.max(probs, dim=0)

            idx = int(idx_v.item())
            conf = float(conf_v.item())
            conf_clamped = round(min(1.0, max(0.0, conf)), 4)

            # Confidence gating: if we're not confident, return "uncertain" and let
            # the orchestrator decide how to fall back (heuristic and/or LLM).
            if conf_clamped < float(CNN_CONF_THRESHOLD):
                return {
                    "label": "uncertain",
                    "confidence": conf_clamped,
                    "severity": "unknown",
                    "source": "cnn_low_confidence",
                    # Back-compat with existing pipeline fields:
                    "damage_type": "unknown",
                    "backend": "cnn_mobilenetv2",
                    "signals": {
                        "cnn_label": "uncertain",
                        "cnn_confidence": conf_clamped,
                        "cnn_used": False,
                        "fallback_reason": "cnn_low_confidence",
                        "conf_threshold": float(CNN_CONF_THRESHOLD),
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    },
                }
            labels_any = ImageCNNService._idx_to_class or ["no_damage", "minor_crack", "major_crack"]
            label_s = labels_any[idx] if 0 <= idx < len(labels_any) else "minor_crack"
            label: Label = (
                label_s  # type: ignore[assignment]
                if label_s in ("no_damage", "minor_crack", "major_crack")
                else "minor_crack"
            )
            severity: Severity = self._severity_from_label(label)

            return {
                "label": label,
                "confidence": conf_clamped,
                "severity": severity,
                # Back-compat with existing pipeline fields:
                "damage_type": label,
                "backend": "cnn_mobilenetv2",
                "signals": {
                    "cnn_label": label,
                    "cnn_confidence": conf_clamped,
                    "cnn_used": True,
                    "conf_threshold": float(CNN_CONF_THRESHOLD),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                },
            }
        except Exception as exc:
            # Safe fallback to heuristic path.
            logger.warning(
                "image_cnn_inference_failed",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
            out = self._fallback.analyze(image_bytes)
            sev = str(out.get("severity") or "low").strip().lower()
            if sev not in ("low", "medium", "high"):
                sev = "medium"
            return {
                "label": "unknown",
                "confidence": float(out.get("confidence") or 0.0),
                "severity": sev,
                "damage_type": str(out.get("damage_type") or "unknown"),
                "backend": "heuristic_fallback",
                "signals": {
                    **(out.get("signals") if isinstance(out.get("signals"), dict) else {}),
                    "fallback_reason": "cnn_failed",
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                },
            }


__all__ = ["ImageCNNService"]
