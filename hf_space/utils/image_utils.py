from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageSignals:
    brightness: float
    edge_density: float
    blur_score: float


def load_pil(image: Any) -> Image.Image:
    """
    Accepts Gradio's image input (PIL.Image | numpy array | path-like) and returns PIL.Image RGB.
    """
    if image is None:
        raise ValueError("image is None")
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    else:
        pil = Image.open(image)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return pil


def resize_rgb(pil: Image.Image, size: int = 224) -> Image.Image:
    return pil.resize((size, size), resample=Image.BILINEAR)


def _to_gray_np(pil: Image.Image) -> np.ndarray:
    gray = pil.convert("L")
    return np.asarray(gray, dtype=np.float32) / 255.0


def _neighbor_edge_density(gray01: np.ndarray) -> float:
    """
    Fast edge proxy using neighbor differences (no cv2 / scipy).
    """
    dx = np.abs(gray01[:, 1:] - gray01[:, :-1])
    dy = np.abs(gray01[1:, :] - gray01[:-1, :])
    # Threshold tuned for 0..1 grayscale
    edge_px = (dx > 0.10).mean() * 0.5 + (dy > 0.10).mean() * 0.5
    return float(edge_px)


def _blur_score(gray01: np.ndarray) -> float:
    """
    Lower score => blurrier. Uses mean absolute neighbor difference as a cheap sharpness proxy.
    """
    dx = np.abs(gray01[:, 1:] - gray01[:, :-1]).mean()
    dy = np.abs(gray01[1:, :] - gray01[:-1, :]).mean()
    return float(0.5 * (dx + dy))


def extract_signals(pil: Image.Image) -> ImageSignals:
    gray01 = _to_gray_np(pil)
    brightness = float(gray01.mean())
    edge_density = _neighbor_edge_density(gray01)
    blur = _blur_score(gray01)
    return ImageSignals(brightness=brightness, edge_density=edge_density, blur_score=blur)


def heuristic_damage_label(signals: ImageSignals) -> tuple[str, str, float]:
    """
    Returns (label, severity, confidence) for demo purposes.

    Intuition:
    - More edges (high-frequency content) often correlates with cracks/noise.
    - Extremely low edges suggests no_damage.
    - Confidence is conservative and bounded.
    """
    e = signals.edge_density
    if e < 0.04:
        return ("no_damage", "low", float(np.clip(1.0 - (e / 0.04), 0.55, 0.95)))
    if e < 0.10:
        conf = float(np.clip((e - 0.04) / 0.06, 0.55, 0.90))
        return ("minor_crack", "medium", conf)
    conf = float(np.clip((e - 0.10) / 0.20, 0.60, 0.92))
    return ("major_crack", "high", conf)


def pil_to_torch_tensor(pil: Image.Image, size: int = 224):
    """
    Converts PIL to normalized tensor suitable for MobileNetV2-style models.
    Torch is imported lazily to keep startup fast.
    """
    import torch
    from torchvision import transforms

    tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    x = tfm(pil).unsqueeze(0)
    return x.to(torch.device("cpu"))

