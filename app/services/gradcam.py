from __future__ import annotations

import base64
import binascii
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class GradCamUnavailable(RuntimeError):
    """Raised when Grad-CAM cannot be produced (missing model/torch, etc.)."""


@dataclass(frozen=True)
class GradCamResult:
    png_bytes: bytes
    saved_path: Optional[str] = None
    label: Optional[str] = None
    confidence: Optional[float] = None


def _decode_claim_image_bytes(image_base64_or_dataurl: str) -> bytes:
    s = str(image_base64_or_dataurl or "").strip()
    if not s:
        raise ValueError("empty image payload")
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1].strip()
    try:
        return base64.b64decode(s, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid base64 image payload") from exc


def _to_pil_rgb(image_bytes: bytes) -> Image.Image:
    bio = io.BytesIO(image_bytes)
    with Image.open(bio) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return im.copy()


def _jet_like_colormap(x01: np.ndarray) -> np.ndarray:
    """Fast, lightweight colormap (no matplotlib). Input in [0,1], output uint8 RGB."""
    x = np.clip(x01.astype(np.float32), 0.0, 1.0)
    # Piecewise "jet-like": blue -> cyan -> yellow -> red
    r = np.clip(1.5 * x - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * x - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * x, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _find_last_conv2d(model: Any) -> Any:
    try:
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise GradCamUnavailable("torch not available") from exc

    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise GradCamUnavailable("no Conv2d layer found for Grad-CAM")
    return last


def _ensure_tmp_dir() -> Path:
    # Requirement says "/tmp/gradcam_<claim_id>.png". We attempt to use /tmp; if that
    # path is unavailable (e.g. Windows), we still generate the image and save best-effort.
    p = Path("/tmp")
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        import tempfile

        return Path(tempfile.gettempdir())


def generate_gradcam_overlay_png(
    *,
    claim_id: str,
    image_bytes: bytes,
    alpha: float = 0.45,
) -> GradCamResult:
    """Generate Grad-CAM overlay PNG for the fine-tuned CNN model.

    CPU-compatible, best-effort. Raises GradCamUnavailable when Grad-CAM cannot be computed.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise GradCamUnavailable("torch not available") from exc

    from app.services.image_cnn_service import ImageCNNService

    svc = ImageCNNService(device="cpu")
    loaded = svc._lazy_load()
    if loaded is None:
        raise GradCamUnavailable("cnn model unavailable (missing torch/weights)")
    model, tfm, device = loaded
    if device is None:
        device = torch.device("cpu")

    im = _to_pil_rgb(image_bytes)
    w0, h0 = im.size

    target_layer = _find_last_conv2d(model)
    activations = None
    gradients = None

    def _fwd_hook(_m, _inp, out):
        nonlocal activations
        activations = out

    def _bwd_hook(_m, _gin, gout):
        nonlocal gradients
        gradients = gout[0] if isinstance(gout, (tuple, list)) and gout else gout

    h1 = target_layer.register_forward_hook(_fwd_hook)
    try:
        # Prefer full backward hook when available; fall back otherwise.
        h2 = (
            target_layer.register_full_backward_hook(_bwd_hook)
            if hasattr(target_layer, "register_full_backward_hook")
            else target_layer.register_backward_hook(_bwd_hook)  # pragma: no cover
        )
    except Exception:  # pragma: no cover
        h2 = target_layer.register_backward_hook(_bwd_hook)

    try:
        x = tfm(im).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True) if hasattr(model, "zero_grad") else None
        with torch.enable_grad():
            logits = model(x)
            if logits is None:
                raise GradCamUnavailable("model returned no logits")
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf_v, idx_v = torch.max(probs, dim=0)
            idx = int(idx_v.item())
            conf = float(conf_v.item())
            score = logits[0, idx]
            score.backward(retain_graph=False)

        if activations is None or gradients is None:
            raise GradCamUnavailable("failed to capture activations/gradients")

        acts = activations.detach()
        grads = gradients.detach()
        if acts.ndim != 4 or grads.ndim != 4:
            raise GradCamUnavailable("unexpected tensor shapes for Grad-CAM")

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=False)  # (1,H,W)
        cam = torch.relu(cam)
        cam_min = float(cam.min().item())
        cam_max = float(cam.max().item())
        cam01 = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam_np = cam01.squeeze(0).cpu().numpy().astype(np.float32)

        cam_img = Image.fromarray((_jet_like_colormap(cam_np)), mode="RGB")
        cam_img = cam_img.resize((w0, h0), resample=Image.Resampling.BILINEAR)

        base = np.asarray(im, dtype=np.float32)
        heat = np.asarray(cam_img, dtype=np.float32)
        a = float(np.clip(alpha, 0.05, 0.85))
        blended = (base * (1.0 - a) + heat * a).clip(0, 255).astype(np.uint8)
        out_im = Image.fromarray(blended, mode="RGB")

        buf = io.BytesIO()
        out_im.save(buf, format="PNG", optimize=True)
        png = buf.getvalue()

        saved_path = None
        try:
            out_dir = _ensure_tmp_dir()
            # Prefer the exact required filename when /tmp is available.
            if str(out_dir).replace("\\", "/") == "/tmp":
                out_path = out_dir / f"gradcam_{claim_id}.png"
            else:
                out_path = out_dir / f"gradcam_{claim_id}.png"
            out_path.write_bytes(png)
            saved_path = str(out_path)
        except Exception:
            saved_path = None

        labels = getattr(ImageCNNService, "_idx_to_class", None) or ["no_damage", "minor_crack", "major_crack"]
        label = labels[idx] if isinstance(labels, list) and 0 <= idx < len(labels) else str(idx)
        return GradCamResult(png_bytes=png, saved_path=saved_path, label=str(label), confidence=conf)
    finally:
        try:
            h1.remove()
        except Exception:
            pass
        try:
            h2.remove()
        except Exception:
            pass


__all__ = [
    "GradCamUnavailable",
    "GradCamResult",
    "generate_gradcam_overlay_png",
    "_decode_claim_image_bytes",
]

