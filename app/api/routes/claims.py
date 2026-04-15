from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from app.agents.orchestrator import InsurFlowOrchestrator, get_insurflow_orchestrator
from app.api.claim_multipart import parse_claim_http_request
from app.core.dependencies import get_vector_store
from app.models.schemas import (
    ClaimImagePreviewResponse,
    ClaimListItem,
    ClaimProcessResponse,
    ClaimReviewRequest,
    DecisionMetadata,
)
from app.services.metrics import metrics
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["claims"])


@router.post("/claims", response_model=ClaimProcessResponse)
async def create_claim(
    request: Request,
    orchestrator: InsurFlowOrchestrator = Depends(get_insurflow_orchestrator),
) -> ClaimProcessResponse:
    payload = await parse_claim_http_request(request)
    result = await orchestrator.process_claim(payload)
    return ClaimProcessResponse.model_validate(result)


@router.get("/claims", response_model=list[ClaimListItem])
async def list_claims(
    vector_store: VectorStore = Depends(get_vector_store),
) -> list[ClaimListItem]:
    rows = vector_store.list_claims(limit=200, offset=0)
    out: list[ClaimListItem] = []
    for row in rows:
        meta = row.get("metadata") or {}
        entities = None
        ej = meta.get("entities_json")
        if isinstance(ej, str) and ej.strip():
            try:
                entities = json.loads(ej)
            except json.JSONDecodeError:
                entities = None
        rs_raw = meta.get("review_status") or meta.get("reviewed_action")
        rs = rs_raw if rs_raw in ("APPROVED", "REJECTED") else None
        hi_raw = str(meta.get("has_image") or "").strip().lower()
        has_image = hi_raw in ("1", "true", "yes")
        llm_raw = str(meta.get("llm_used") or "").strip().lower()
        llm_used = llm_raw in ("1", "true", "yes")
        fb_raw = str(meta.get("fallback_used") or "").strip().lower()
        fallback_used = fb_raw in ("1", "true", "yes")
        ds = str(meta.get("decision_source") or "").strip().lower() or None
        if ds not in ("rule", "llm", "fallback"):
            ds = None
        fraud_signal = str(meta.get("fraud_signal") or "").strip() or None
        cnn_used_raw = str(meta.get("cnn_used") or "").strip().lower()
        cnn_used = cnn_used_raw in ("1", "true", "yes")
        cnn_label = str(meta.get("cnn_label") or "").strip()
        try:
            cnn_confidence = float(meta.get("cnn_confidence") or 0.0)
        except (TypeError, ValueError):
            cnn_confidence = 0.0
        cnn_confidence = min(1.0, max(0.0, cnn_confidence))
        cnn_severity = str(meta.get("cnn_severity") or "").strip()

        pipeline_obj = None
        pj = meta.get("pipeline_json")
        if isinstance(pj, str) and pj.strip():
            try:
                pipeline_obj = DecisionMetadata.model_validate_json(pj)
            except Exception:
                pipeline_obj = None
        out.append(
            ClaimListItem(
                claim_id=str(row.get("claim_id") or ""),
                claim_description=str(row.get("claim_description") or ""),
                fraud_score=float(meta.get("fraud_score") or 0.0),
                decision=meta.get("decision"),
                confidence=meta.get("confidence"),
                explanation=meta.get("explanation"),
                review_status=rs,
                hitl_needed=bool(meta.get("hitl_needed") or False),
                reviewed_action=meta.get("reviewed_action"),
                reviewed_at=meta.get("reviewed_at"),
                reviewed_by=meta.get("reviewed_by"),
                entities=entities,
                has_image=has_image,
                image_damage_type=str(meta.get("image_damage_type") or ""),
                image_severity=str(meta.get("image_severity") or ""),
                llm_used=llm_used,
                fallback_used=fallback_used,
                decision_source=ds,
                metadata=pipeline_obj,
                fraud_signal=fraud_signal,
                cnn_used=cnn_used,
                cnn_label=cnn_label,
                cnn_confidence=cnn_confidence,
                cnn_severity=cnn_severity,
            )
        )
    return out


@router.get("/claims/{claim_id}/image-preview", response_model=ClaimImagePreviewResponse)
async def get_claim_image_preview(
    claim_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
) -> ClaimImagePreviewResponse:
    existing = vector_store.get_claim(claim_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")
    meta = existing.get("metadata") or {}
    b64 = str(meta.get("image_preview_base64") or "").strip()
    if not b64:
        raise HTTPException(status_code=404, detail="No image preview stored for this claim.")
    return ClaimImagePreviewResponse(claim_id=str(existing.get("claim_id") or claim_id), image_base64=b64)


@router.get("/claims/{claim_id}/gradcam")
async def get_claim_gradcam(
    claim_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
) -> Response:
    """Return a Grad-CAM heatmap overlay PNG for the stored claim image.

    Best-effort, CPU-safe. If the CNN stack/weights are unavailable, returns 503.
    """
    existing = vector_store.get_claim(claim_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")
    meta = existing.get("metadata") or {}
    b64 = str(meta.get("image_preview_base64") or "").strip()
    if not b64:
        raise HTTPException(status_code=404, detail="No image preview stored for this claim.")

    try:
        from app.services.gradcam import GradCamUnavailable, _decode_claim_image_bytes, generate_gradcam_overlay_png

        raw = _decode_claim_image_bytes(b64)
        out = generate_gradcam_overlay_png(claim_id=claim_id, image_bytes=raw)
        headers = {}
        if out.saved_path:
            headers["X-Gradcam-Path"] = out.saved_path
        if out.label:
            headers["X-Gradcam-Label"] = str(out.label)
        if out.confidence is not None:
            headers["X-Gradcam-Confidence"] = f"{float(out.confidence):.6f}"
        return Response(content=out.png_bytes, media_type="image/png", headers=headers)
    except GradCamUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Grad-CAM unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("claim_gradcam_failed", extra={"claim_id": claim_id})
        raise HTTPException(status_code=500, detail=f"Failed to generate Grad-CAM: {type(exc).__name__}") from exc


@router.post("/claims/{claim_id}/review")
async def review_claim(
    claim_id: str,
    body: ClaimReviewRequest,
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    action = body.action
    try:
        existing = vector_store.get_claim(claim_id)
        if not existing:
            raise KeyError(f"Claim '{claim_id}' not found.")

        meta = dict(existing.get("metadata") or {})
        meta.setdefault("explanation", "")
        if not str(meta.get("explanation") or "").strip():
            meta["explanation"] = json.dumps(
                {
                    "summary": "Legacy record; explanation was missing.",
                    "key_factors": ["Restored when saving human review."],
                    "similar_case_reference": "",
                },
                ensure_ascii=False,
            )
        meta.setdefault("entities_json", "{}")
        meta.setdefault("fraud_score", 0.0)
        meta.setdefault("decision", "")
        meta.setdefault("confidence", 0.0)
        meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        meta.setdefault("review_status", "")
        meta.setdefault("case_status", "NEW")
        meta.setdefault("assigned_to", "")
        meta.setdefault("assigned_at", "")
        meta.setdefault("image_damage_type", "")
        meta.setdefault("image_severity", "")
        meta.setdefault("has_image", "")
        meta.setdefault("image_preview_base64", "")
        meta.setdefault("llm_used", "")
        meta.setdefault("decision_source", "")
        meta.setdefault("updated_at", meta.get("timestamp") or datetime.now(timezone.utc).isoformat())
        now = datetime.now(timezone.utc).isoformat()
        meta.update(
            {
                "review_status": action,
                "reviewed_action": action,
                "reviewed_by": body.reviewed_by or "human_reviewer",
                "reviewed_at": now,
                "hitl_needed": False,
                "hitl_reason": "",
                "updated_at": now,
            }
        )
        vector_store.store_claim(
            claim_id=str(existing.get("claim_id") or claim_id),
            claim_description=str(existing.get("claim_description") or ""),
            embedding=list(existing.get("embedding") or []),
            metadata=meta,
        )
        metrics.record_review()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("claim_review_failed", extra={"claim_id": claim_id})
        raise HTTPException(status_code=500, detail=f"Failed to review claim: {type(exc).__name__}") from exc
    return {"ok": True, "claim_id": claim_id, "action": action}
