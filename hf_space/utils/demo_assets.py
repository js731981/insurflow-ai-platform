"""Generate bundled demo images under assets/demo_images/."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def ensure_demo_images(root: Path) -> Path:
    demo_dir = root / "assets" / "demo_images"
    demo_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    h, w = 512, 768

    def write_if_missing(name: str, pil: Image.Image) -> None:
        p = demo_dir / name
        if not p.exists():
            pil.save(p, format="JPEG", quality=88)

    low = np.ones((h, w, 3), dtype=np.uint8) * 235
    low += (rng.random((h, w, 3)) * 20).astype(np.uint8)
    pil_low = Image.fromarray(low, mode="RGB")
    d = ImageDraw.Draw(pil_low)
    d.rectangle([40, 40, w - 40, h - 40], outline=(120, 130, 150), width=3)
    d.line([(140, 220), (380, 300)], fill=(70, 80, 100), width=2)
    d.text((50, 50), "Demo: low damage (valid-style)", fill=(30, 41, 59))
    write_if_missing("low_damage_valid.jpg", pil_low)

    smooth = np.ones((h, w, 3), dtype=np.uint8) * 248
    smooth[:, :] += np.array([5, 7, 10], dtype=np.uint8)
    pil_s = Image.fromarray(smooth, mode="RGB")
    d2 = ImageDraw.Draw(pil_s)
    d2.rectangle([60, 60, w - 60, h - 60], outline=(180, 190, 210), width=2)
    d2.text((60, 70), "Demo: no visible damage", fill=(30, 41, 59))
    write_if_missing("no_damage_fraud.jpg", pil_s)

    maj = np.ones((h, w, 3), dtype=np.uint8) * 210
    maj += (rng.random((h, w, 3)) * 45).astype(np.uint8)
    pil_m = Image.fromarray(maj, mode="RGB")
    d3 = ImageDraw.Draw(pil_m)
    for i in range(8):
        d3.line([(80 + i * 40, 100), (200 + i * 35, 400)], fill=(40, 20, 20), width=4)
    d3.line([(100, 120), (600, 380)], fill=(180, 30, 30), width=6)
    d3.text((70, 60), "Demo: major damage pattern", fill=(15, 23, 42))
    write_if_missing("major_damage_high.jpg", pil_m)

    return demo_dir
