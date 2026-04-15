"""
HTTP client for the Insurance AI backend. All remote calls from the HF demo go through here.
"""
from __future__ import annotations

import io
import os
import time
from typing import Any, Optional
from urllib.parse import quote

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=15.0)


def get_base_url() -> str:
    raw = os.environ.get("INSURANCE_API_BASE_URL") or os.environ.get("BACKEND_URL") or os.environ.get("API_BASE_URL") or ""
    return str(raw).strip().rstrip("/")


def is_configured() -> bool:
    return bool(get_base_url())


class ApiClientError(RuntimeError):
    pass


def _client() -> httpx.Client:
    base = get_base_url()
    if not base:
        raise ApiClientError("Backend URL not configured. Set INSURANCE_API_BASE_URL (or BACKEND_URL).")
    return httpx.Client(base_url=base, timeout=DEFAULT_TIMEOUT)


def get_analytics_summary() -> dict[str, Any]:
    """GET /analytics/summary"""
    with _client() as c:
        r = c.get("/analytics/summary")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else {}


def submit_claim_multipart(
    *,
    claim_id: str,
    description: str,
    claim_amount: float,
    policy_limit: float,
    image_bytes: Optional[bytes],
    image_filename: str = "claim.jpg",
) -> tuple[dict[str, Any], float]:
    """
    POST /claims as multipart/form-data (flat fields + optional image).

    Returns (response_json, elapsed_ms).
    """
    t0 = time.perf_counter()
    # httpx requires `files=` (not `data=`) so the request is multipart/form-data as FastAPI expects.
    parts: list[tuple[str, Any]] = [
        ("claim_id", (None, claim_id)),
        ("description", (None, description or "")),
        ("claim_amount", (None, str(claim_amount))),
        ("policy_limit", (None, str(policy_limit))),
    ]
    if image_bytes:
        parts.append(("image", (image_filename, io.BytesIO(image_bytes), "image/jpeg")))
    with _client() as c:
        r = c.post("/claims", files=parts)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code >= 400:
            detail = r.text[:2000]
            raise ApiClientError(f"POST /claims failed ({r.status_code}): {detail}")
        data = r.json()
        if not isinstance(data, dict):
            raise ApiClientError("POST /claims returned non-JSON object")
        return data, elapsed_ms


def fetch_gradcam_png(claim_id: str) -> Optional[bytes]:
    """GET /claims/{claim_id}/gradcam — returns PNG bytes or None on 404/503."""
    cid = str(claim_id or "").strip()
    if not cid or not is_configured():
        return None
    try:
        with _client() as c:
            r = c.get(f"/claims/{quote(cid, safe='')}/gradcam")
    except (ApiClientError, httpx.HTTPError):
        return None
    if r.status_code == 200:
        return r.content
    return None


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _flatten_explanation_block(raw: Any) -> str:
    if isinstance(raw, dict):
        lines: list[str] = []
        s = str(raw.get("summary") or "").strip()
        if s:
            lines.append(s)
        kf = raw.get("key_factors")
        if isinstance(kf, list):
            for x in kf[:8]:
                t = str(x).strip()
                if t:
                    lines.append(f"- {t}")
        ref = str(raw.get("similar_case_reference") or "").strip()
        if ref:
            lines.append(f"Similar cases: {ref}")
        return "\n".join(lines) if lines else ""
    if isinstance(raw, str):
        return raw.strip()
    return ""


def parse_claim_api_json(
    data: dict[str, Any],
    *,
    client_latency_ms: float,
    had_image_upload: bool,
    claim_amount: float,
    policy_limit: float,
) -> dict[str, Any]:
    """
    Normalize POST /claims JSON into the shape consumed by the HF demo UI (no HTML).
    """
    fraud = data.get("agent_outputs", {}).get("fraud") if isinstance(data.get("agent_outputs"), dict) else {}
    if not isinstance(fraud, dict):
        fraud = {}
    policy = data.get("agent_outputs", {}).get("policy") if isinstance(data.get("agent_outputs"), dict) else {}
    if not isinstance(policy, dict):
        policy = {}
    decision = data.get("agent_outputs", {}).get("decision") if isinstance(data.get("agent_outputs"), dict) else {}
    if not isinstance(decision, dict):
        decision = {}
    image_block = data.get("agent_outputs", {}).get("image") if isinstance(data.get("agent_outputs"), dict) else {}
    if not isinstance(image_block, dict):
        image_block = {}

    md = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    llm_status = str(md.get("llm_status") or "skipped").strip().lower()
    if llm_status not in ("used", "skipped", "failed"):
        llm_status = "skipped"

    fraud_score = _f(fraud.get("fraud_score"), 0.0)
    fraud_score = max(0.0, min(1.0, fraud_score))

    w = decision.get("fraud_fusion_weights") if isinstance(decision.get("fraud_fusion_weights"), dict) else {}
    w_llm = _f(w.get("llm"), 0.33)
    w_dl = _f(w.get("dl"), 0.33)
    w_img = _f(w.get("image"), 0.33)
    f_llm = _f(decision.get("fraud_score_llm"), fraud_score)
    img_score = decision.get("image_severity_score")
    img_f = _f(img_score, 0.0) if img_score is not None else 0.0
    dl_raw = decision.get("fraud_probability_dl")
    dl_f = _f(dl_raw, 0.0) if dl_raw is not None else 0.0

    policy_ok = bool(policy.get("policy_valid", True))
    rules_push = (0.35 if not policy_ok else 0.08) + w_dl * dl_f
    cnn_floor = 0.15 if str(data.get("cnn_label") or "").lower() not in ("", "unknown", "n/a") else 0.0
    cnn_push = (w_img * max(img_f, cnn_floor)) if had_image_upload else 0.0
    llm_push = w_llm * (f_llm - 0.5) * 2.0

    cnn_used_api = bool(data.get("cnn_used"))

    cnn_ms = _f(image_block.get("processing_time_ms"), 0.0)
    total_ms = max(client_latency_ms, 0.0)
    llm_ms = max(0.0, total_ms - cnn_ms - 12.0)  # small overhead estimate when server omits LLM timing

    pipeline: list[dict[str, str]] = [
        {"id": "image", "label": "Image", "status": "used" if had_image_upload else "skipped"},
        {"id": "cnn", "label": "CNN", "status": "used" if (had_image_upload and cnn_used_api) else "skipped"},
        {"id": "rules", "label": "Rules", "status": "used"},
        {
            "id": "llm",
            "label": "LLM",
            "status": llm_status if llm_status in ("used", "skipped", "failed") else "skipped",
        },
        {"id": "decision", "label": "Decision", "status": "used"},
    ]

    expl_text = _flatten_explanation_block(fraud.get("explanation"))

    fs = str(data.get("fraud_signal") or "").strip() or None
    try:
        pol_lim = float(policy_limit)
    except (TypeError, ValueError):
        pol_lim = 0.0
    try:
        c_amt = float(claim_amount)
    except (TypeError, ValueError):
        c_amt = 0.0
    amt_high = bool((pol_lim > 0 and c_amt > 0.85 * pol_lim) or c_amt > 7500.0)

    inc_msgs: list[str] = []
    if fs == "image_text_mismatch":
        inc_msgs.append("Image suggests no/minimal damage but the narrative describes substantial damage.")
    if amt_high:
        if pol_lim > 0 and c_amt > 0.85 * pol_lim:
            inc_msgs.append("Claim amount is unusually high relative to the stated policy limit.")
        if c_amt > 7500.0:
            inc_msgs.append("Claim amount exceeds typical micro-claim thresholds.")

    return {
        "source": "api",
        "claim_id": str(data.get("claim_id") or ""),
        "fraud_score": fraud_score,
        "decision": str(data.get("decision") or ""),
        "cnn_label": str(data.get("cnn_label") or "unknown"),
        "cnn_confidence": _f(data.get("cnn_confidence"), 0.0),
        "cnn_severity": str(data.get("cnn_severity") or "unknown"),
        "explanation": expl_text,
        "fraud_signal": fs,
        "breakdown": {"cnn": round(cnn_push, 3), "rules": round(rules_push, 3), "llm": round(llm_push, 3)},
        "pipeline": pipeline,
        "latency_ms": {"total": round(total_ms, 1), "cnn": round(cnn_ms, 1), "llm": round(llm_ms, 1)},
        "metadata": md,
        "agent_outputs": data.get("agent_outputs") or {},
        "inconsistent_claim": bool(inc_msgs),
        "inconsistency_messages": inc_msgs,
    }
