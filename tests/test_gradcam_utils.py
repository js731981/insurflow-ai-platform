from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from app.services.gradcam import GradCamUnavailable, _decode_claim_image_bytes, generate_gradcam_overlay_png


def _png_bytes() -> bytes:
    im = Image.new("RGB", (32, 32), color="white")
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_claim_image_bytes_accepts_plain_base64():
    raw = _png_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    out = _decode_claim_image_bytes(b64)
    assert out == raw


def test_decode_claim_image_bytes_accepts_data_url():
    raw = _png_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    out = _decode_claim_image_bytes(data_url)
    assert out == raw


def test_generate_gradcam_fails_safe_when_model_unavailable(monkeypatch):
    # Avoid coupling this unit test to torch/weights availability.
    from app import services as _  # noqa: F401
    from app.services import image_cnn_service as mod

    monkeypatch.setattr(mod.ImageCNNService, "_lazy_load", lambda self: None)
    with pytest.raises(GradCamUnavailable):
        generate_gradcam_overlay_png(claim_id="C-1", image_bytes=_png_bytes())

