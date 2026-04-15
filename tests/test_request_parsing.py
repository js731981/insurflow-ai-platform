"""Tests for multipart / JSON claim request parsing."""

from __future__ import annotations

import asyncio
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from PIL import Image
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from app.api.claim_multipart import parse_claim_http_request


def _minimal_claim_dict(
    *,
    claim_id: str = "MIC-TEST-001",
    claim_amount: float = 75.5,
    policy_limit: float = 500.0,
    description: str = "Synthetic claim for tests.",
) -> dict:
    return {
        "claim_id": claim_id,
        "claim_amount": claim_amount,
        "policy_limit": policy_limit,
        "description": description,
    }


def _png_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _multipart_request(form_get):
    req = MagicMock(spec=Request)
    req.headers = {"content-type": "multipart/form-data; boundary=----testboundary"}
    form = SimpleNamespace(get=form_get)
    req.form = AsyncMock(return_value=form)
    return req


def _run(coro):
    return asyncio.run(coro)


def test_multipart_valid_with_image_extracts_fields():
    png = _png_bytes()
    bio = BytesIO(png)
    bio.seek(0)
    upload = UploadFile(bio, filename="proof.png")

    claim = _minimal_claim_dict(
        claim_id="C-42",
        claim_amount=99.0,
        description="Screen damage",
    )

    def form_get(key, default=None):
        if key == "claim":
            return json.dumps(claim)
        if key == "image":
            return upload
        return default

    payload = _run(parse_claim_http_request(_multipart_request(form_get)))
    assert payload["claim_id"] == "C-42"
    assert payload["claim_amount"] == 99.0
    assert payload["description"] == "Screen damage"
    assert payload["policy_limit"] == 500.0
    assert payload["_image_bytes"] == png


def test_multipart_valid_without_image_no_image_bytes():
    claim = _minimal_claim_dict()

    def form_get(key, default=None):
        if key == "claim":
            return json.dumps(claim)
        return default

    payload = _run(parse_claim_http_request(_multipart_request(form_get)))
    assert "_image_bytes" not in payload
    assert payload["claim_id"] == claim["claim_id"]


def test_multipart_missing_claim_field():
    def form_get(key, default=None):
        return default

    with pytest.raises(HTTPException) as ei:
        _run(parse_claim_http_request(_multipart_request(form_get)))
    assert ei.value.status_code == 422
    assert "claim" in (ei.value.detail or "").lower()


def test_multipart_claim_not_string():
    bio = BytesIO(_png_bytes())
    bio.seek(0)
    upload = UploadFile(bio, filename="x.png")

    def form_get(key, default=None):
        if key == "claim":
            return upload
        return default

    with pytest.raises(HTTPException) as ei:
        _run(parse_claim_http_request(_multipart_request(form_get)))
    assert ei.value.status_code == 422
    assert "string" in (ei.value.detail or "").lower()


def test_multipart_invalid_claim_schema_missing_required():
    def form_get(key, default=None):
        if key == "claim":
            return json.dumps({"claim_id": "only-id"})
        return default

    with pytest.raises(ValidationError):
        _run(parse_claim_http_request(_multipart_request(form_get)))


def test_multipart_invalid_json_in_claim_field():
    def form_get(key, default=None):
        if key == "claim":
            return "{not-json"
        return default

    with pytest.raises(HTTPException) as ei:
        _run(parse_claim_http_request(_multipart_request(form_get)))
    assert ei.value.status_code == 422


def test_multipart_image_field_wrong_type():
    claim = _minimal_claim_dict()

    def form_get(key, default=None):
        if key == "claim":
            return json.dumps(claim)
        if key == "image":
            return "not-a-file"
        return default

    with pytest.raises(HTTPException) as ei:
        _run(parse_claim_http_request(_multipart_request(form_get)))
    assert ei.value.status_code == 422
    assert "file" in (ei.value.detail or "").lower()


def test_multipart_empty_image_upload():
    claim = _minimal_claim_dict()
    bio = BytesIO(b"")
    upload = UploadFile(bio, filename="empty.png")

    def form_get(key, default=None):
        if key == "claim":
            return json.dumps(claim)
        if key == "image":
            return upload
        return default

    with pytest.raises(HTTPException) as ei:
        _run(parse_claim_http_request(_multipart_request(form_get)))
    assert ei.value.status_code == 422


def test_json_body_valid():
    claim = _minimal_claim_dict()
    req = MagicMock(spec=Request)
    req.headers = {"content-type": "application/json"}
    req.json = AsyncMock(return_value=claim)
    payload = _run(parse_claim_http_request(req))
    assert payload["claim_id"] == claim["claim_id"]
    assert "_image_bytes" not in payload


def test_multipart_flat_fields_with_file_alias():
    png = _png_bytes()
    bio = BytesIO(png)
    bio.seek(0)
    upload = UploadFile(bio, filename="damage.jpg")

    def form_get(key, default=None):
        if key == "claim_id":
            return "FLAT-1"
        if key == "claim_amount":
            return "12.5"
        if key == "policy_limit":
            return "400"
        if key == "description":
            return "Hail on roof"
        if key == "file":
            return upload
        return default

    payload = _run(parse_claim_http_request(_multipart_request(form_get)))
    assert payload["claim_id"] == "FLAT-1"
    assert payload["claim_amount"] == 12.5
    assert payload["policy_limit"] == 400.0
    assert payload["description"] == "Hail on roof"
    assert payload["_image_bytes"] == png
