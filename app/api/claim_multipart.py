from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import HTTPException, Request, UploadFile

from app.models.schemas import ClaimRequest

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 12 * 1024 * 1024
_MAX_CLAIM_JSON_BYTES = 256 * 1024


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in claim field: {exc}") from exc
    if not isinstance(obj, dict):
        raise HTTPException(status_code=422, detail="claim JSON must be an object")
    return obj


def _scalar_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


async def parse_claim_http_request(request: Request) -> dict[str, Any]:
    """Parse JSON body or multipart/form-data into a claim dict.

    Multipart supports:

    * ``claim`` (JSON string) + optional file field ``image`` (legacy), or
    * Flat fields ``claim_id``, ``claim_amount``, ``policy_limit``, optional ``description``, and optional
      file field ``file`` or ``image`` (dashboard / simple clients).
    """
    ct = request.headers.get("content-type") or ""
    if "multipart/form-data" in ct.lower():
        form = await request.form()
        claim_raw = form.get("claim")
        if claim_raw is None:
            cid = _scalar_str(form.get("claim_id"))
            if not cid:
                raise HTTPException(
                    status_code=422,
                    detail="Multipart must include either field 'claim' (JSON string) or field 'claim_id' with amounts.",
                )
            amt_raw = form.get("claim_amount")
            lim_raw = form.get("policy_limit")
            try:
                claim_amount = float(amt_raw)  # type: ignore[arg-type]
                policy_limit = float(lim_raw)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail="claim_amount and policy_limit must be valid numbers.",
                ) from exc
            desc = _scalar_str(form.get("description"))
            payload = {
                "claim_id": cid,
                "claim_amount": claim_amount,
                "policy_limit": policy_limit,
                "description": desc or None,
            }
            ClaimRequest.model_validate(payload)
        else:
            if not isinstance(claim_raw, str):
                raise HTTPException(status_code=422, detail="Multipart field 'claim' must be a string of JSON")
            raw = claim_raw.strip()
            if len(raw.encode("utf-8")) > _MAX_CLAIM_JSON_BYTES:
                raise HTTPException(status_code=413, detail="claim JSON too large")
            payload = _safe_json_loads(raw)
            ClaimRequest.model_validate(payload)

        img = form.get("file")
        if img is None:
            img = form.get("image")
        if img is not None:
            if not isinstance(img, UploadFile):
                raise HTTPException(status_code=422, detail="Multipart file field 'file' or 'image' must be a file upload")
            filename = (img.filename or "").lower()
            ctype = (img.content_type or "").lower()
            if ctype and ctype not in ("image/jpeg", "image/jpg", "image/png", "image/pjpeg", "image/x-png"):
                if not any(filename.endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
                    raise HTTPException(
                        status_code=415,
                        detail="Unsupported image type; use image/jpeg or image/png.",
                    )
            try:
                data = await img.read()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to read upload: {type(exc).__name__}") from exc
            if not data:
                # Some browsers/clients may submit a zero-byte file entry (e.g. cancelled picker).
                # Treat as "no image" rather than failing the whole claim submission.
                logger.info(
                    "multipart_claim_parsed",
                    extra={"has_image": False, "empty_upload": True, "filename": img.filename},
                )
                return payload
            if len(data) > _MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Image file too large")
            payload = dict(payload)
            payload["_image_bytes"] = data
            logger.info(
                "multipart_claim_parsed",
                extra={"has_image": True, "image_bytes": len(data), "filename": img.filename},
            )
        else:
            logger.info("multipart_claim_parsed", extra={"has_image": False})
        return payload

    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="JSON body must be an object")
    req = ClaimRequest.model_validate(body)
    return req.model_dump(exclude_none=True)


__all__ = ["parse_claim_http_request"]
