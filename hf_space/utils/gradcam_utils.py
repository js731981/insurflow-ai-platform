"""Demo attention heatmap + overlay (Spaces-safe, no full Grad-CAM dependency)."""
from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np
from PIL import Image


def _heatmap_from_gray(gray: np.ndarray) -> np.ndarray:
    """Map single channel 0..1 to RGB heatmap (dark blue → cyan → yellow → red)."""
    g = np.clip(gray.astype(np.float32), 0.0, 1.0)
    r = np.clip((g - 0.5) * 2.0, 0.0, 1.0)
    gb = np.clip(1.0 - np.abs(g - 0.5) * 2.0, 0.0, 1.0)
    b = np.clip((0.5 - g) * 2.0, 0.0, 1.0)
    rgb = np.stack([r, gb, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def compute_edge_attention_map(pil_rgb: Image.Image, blur_ksize: int = 5) -> Image.Image:
    """
    Cheap pseudo–Class Activation Map from luminance gradients (visual demo only).
    Returns RGB heatmap image same size as input.
    """
    g = np.asarray(pil_rgb.convert("L"), dtype=np.float32) / 255.0
    gx = np.abs(g[:, 1:] - g[:, :-1])
    gy = np.abs(g[1:, :] - g[:-1, :])
    h, w = g.shape
    mag = np.zeros((h, w), dtype=np.float32)
    mag[:, 1:] += gx
    mag[1:, :] += gy
    mag[:, 0] += mag[:, 1]
    mag[0, :] += mag[1, :]
    # Cheap smoothing: downscale + upscale (no scipy dependency).
    if blur_ksize > 1:
        im = Image.fromarray(np.asarray(mag, dtype=np.float32), mode="F")
        sw, sh = max(1, w // 10), max(1, h // 10)
        im = im.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        mag = np.asarray(im, dtype=np.float32)
    mag = mag - float(mag.min())
    denom = float(mag.max()) + 1e-6
    mag = mag / denom
    rgb = _heatmap_from_gray(mag)
    return Image.fromarray(rgb, mode="RGB")


def blend_overlay(base: Image.Image, heatmap: Image.Image, opacity: float) -> Image.Image:
    """Alpha-blend heatmap onto base. opacity in 0..1."""
    a = max(0.0, min(1.0, float(opacity)))
    if a <= 0.0:
        return base.convert("RGB")
    b = base.convert("RGB")
    h = heatmap.convert("RGB")
    if h.size != b.size:
        h = h.resize(b.size, Image.BILINEAR)
    bb = np.asarray(b).astype(np.float32)
    hh = np.asarray(h).astype(np.float32)
    out = (bb * (1.0 - a) + hh * a).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def pil_from_gradcam_png(data: bytes) -> Optional[Image.Image]:
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        return im
    except Exception:
        return None


def pil_to_png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def composite_gradcam_view(
    image: Any,
    heat_png_bytes: Optional[bytes],
    opacity_pct: float,
    show: bool,
    *,
    server_overlay: bool = False,
) -> Any:
    """Return PIL for Gradio image; `server_overlay` True when heat is already a colored overlay from backend."""
    if image is None or not show:
        return image
    try:
        from hf_space.utils.image_utils import load_pil

        base = load_pil(image)
    except Exception:
        return image
    op = max(0.0, min(100.0, float(opacity_pct))) / 100.0
    if heat_png_bytes:
        layer = pil_from_gradcam_png(heat_png_bytes)
        if layer is None:
            layer = compute_edge_attention_map(base)
        if server_overlay:
            if layer.size != base.size:
                layer = layer.resize(base.size, Image.BILINEAR)
            return Image.blend(base, layer, alpha=op)
        return blend_overlay(base, layer, op)
    heat = compute_edge_attention_map(base)
    return blend_overlay(base, heat, op)


def attention_preview(image: Any, opacity_pct: float, show: bool):
    """Gradio helper: returns PIL overlay or original."""
    return composite_gradcam_view(image, None, opacity_pct, show, server_overlay=False)
