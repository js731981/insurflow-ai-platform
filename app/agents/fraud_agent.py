from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, NamedTuple, Optional, Tuple, TypedDict

from app.agents.base_agent import BaseAgent
from app.core.config import settings
from app.services.llm_service import LLMService
from app.services.llm.providers.base import LLMProviderError

logger = logging.getLogger(__name__)

_RAW_LOG_MAX_CHARS = 12_000
_MAX_DESCRIPTION_CHARS = 300
_FRAUD_MAX_TOKENS = 300

def _truncate_for_log(text: str, max_chars: int = _RAW_LOG_MAX_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


class FraudParseResult(NamedTuple):
    fraud_score: float
    decision: str
    confidence: float
    explanation: dict[str, Any]
    entities: dict[str, Any]
    ok: bool
    error: Optional[str]


class FraudAgentInput(TypedDict, total=False):
    """Orchestrator-supplied payload (arbitrary claim keys allowed via ``total=False`` + extras).

    ``similar_claims_context`` is produced by the RAG :class:`~app.services.context_builder.ContextBuilder`.
    """

    claim_id: str
    description: str
    claim_amount: float
    policy_limit: float
    currency: str
    product_code: str
    incident_date: str
    policyholder_id: str
    rag_filter_decision: str
    rag_metadata_filter: dict[str, Any]
    similar_claims_context: str
    image_features: dict[str, Any]


def _default_explanation() -> dict[str, Any]:
    """Shown when the model response is missing, non-JSON, or fails schema checks (aligns with DecisionAgent fallback)."""
    return {
        "summary": (
            "Fraud model did not return usable structured output; treat as escalate for manual review."
        ),
        "key_factors": [
            "Response was not valid JSON or omitted required fields (e.g. explanation.summary, two or more key_factors).",
            "Automated triage cannot rely on this output; same outcome as fraud model unavailable.",
        ],
        "similar_case_reference": "",
    }


def _claim_description_for_llm(input_data: dict[str, Any]) -> str:
    """Minimal claim text for the fraud prompt (description only)."""
    desc = str(input_data.get("description") or "").strip()
    if desc:
        return desc[:_MAX_DESCRIPTION_CHARS]
    cid = str(input_data.get("claim_id") or "").strip()
    if cid:
        return f"(No claim description provided; claim_id={cid}.)"[:_MAX_DESCRIPTION_CHARS]
    return "(No claim description provided.)"[:_MAX_DESCRIPTION_CHARS]


def _image_features_one_line(image_features: dict[str, Any]) -> str:
    if not image_features or image_features.get("present") is False:
        return "Image: not provided"
    damage_type = str(image_features.get("damage_type") or "").strip()
    severity = str(image_features.get("severity") or "").strip()
    if damage_type and severity:
        return f"Image: {severity} severity {damage_type}"
    if severity:
        return f"Image: {severity} severity damage"
    if damage_type:
        return f"Image: {damage_type}"
    return "Image: damage"


def _strip_markdown_json_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _extract_json_balanced(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    quote_ch = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_ch:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            quote_ch = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json_object_loose(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _repair_json_candidate(candidate: str) -> str:
    t = candidate.strip()
    if t.startswith("\ufeff"):
        t = t[1:]
    for _ in range(8):
        n = re.sub(r",\s*}", "}", t)
        n = re.sub(r",\s*]", "]", n)
        if n == t:
            break
        t = n
    return t


class FraudAgent(BaseAgent):
    """Fraud signals for micro-insurance claims (LLM-assisted, JSON score + structured explanation)."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
    ) -> None:
        super().__init__()
        self._llm_service = llm_service

    def _build_fraud_prompt(
        self,
        *,
        description: str,
        amount: float,
        image_severity: str,
    ) -> str:
        # Performance-first prompt (small + deterministic). Avoid RAG dumps to prevent timeouts.
        # IMPORTANT: Keep this aligned with the parser's required fields.
        return f"""
You MUST return ONLY valid JSON.

Do NOT include any explanation text.

Return exactly this format:

{{
"fraud_score": number (0 to 1),
"decision": "APPROVE" | "INVESTIGATE" | "REJECT",
"reasons": ["reason1", "reason2"]
}}

Claim:
Description: {description}
Amount: {amount}
Image severity: {image_severity}
""".strip()

    def _build_fixup_prompt(self, *, bad_output: str) -> str:
        clipped = (bad_output or "").strip()
        if len(clipped) > 1500:
            clipped = clipped[:1500]
        return f"""
You returned output that is NOT valid JSON.

Fix it and return ONLY valid JSON, with exactly this format:

{{
"fraud_score": number (0 to 1),
"decision": "APPROVE" | "INVESTIGATE" | "REJECT",
"reasons": ["reason1", "reason2"]
}}

Bad output to fix:
{clipped}
""".strip()

    def _prompt_fields(self, input_data: dict[str, Any]) -> tuple[str, float, str]:
        description = _claim_description_for_llm(input_data)
        try:
            amount = float(input_data.get("claim_amount") or input_data.get("amount") or 0.0)
        except (TypeError, ValueError):
            amount = 0.0

        image_features = input_data.get("image_features")
        if not isinstance(image_features, dict):
            image_features = {"present": False}
        image_severity = str(image_features.get("severity") or "").strip() or "unknown"
        return description, round(float(amount), 2), image_severity

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        claim_id = str(input_data.get("claim_id") or "").strip() or "unknown"
        start_wall_s = time.time()
        start_perf = time.perf_counter()
        image_features = input_data.get("image_features")
        if not isinstance(image_features, dict):
            image_features = {"present": False, "damage_type": "n/a", "severity": "n/a", "confidence": 0.0}

        total_latency_ms = 0
        last_raw = ""
        llm_response_time_ms: int | None = None

        # Stay within the upstream FraudAgent SLA by preventing router-level retries here.
        # The orchestrator already applies its own hard cap; multi-attempt LLM retries can
        # otherwise cause cancellation mid-flight and produce "agent timed out" fallbacks.
        # IMPORTANT: Do not shrink below the LLM timeout floor (>=60s). If the overall
        # request has a tighter SLA, the outer timeout will cancel the in-flight request.
        # Separate, explicit timeout boundaries:
        # - LLM timeout: caps ONLY the network/model call.
        # - Agent timeout: overall budget for prompt build + LLM + parsing (not enforced here; orchestrator should not cancel post-LLM).
        llm_timeout_s = float(settings.llm_timeout_s)
        effective_timeout_s = max(60.0, llm_timeout_s)
        agent_timeout_s = float(settings.fraud_agent_timeout_s)

        description, amount, image_severity = self._prompt_fields(input_data)
        prompt = self._build_fraud_prompt(description=description, amount=amount, image_severity=image_severity)
        gen_kw: dict[str, Any] = {
            # Provider-agnostic cap. OllamaProvider maps `max_tokens` to `options.num_predict`.
            "max_tokens": _FRAUD_MAX_TOKENS,
            "temperature": 0.2,
        }

        prompt_length = len(prompt)
        logger.info(
            "fraud_llm_prompt_built",
            extra={
                "claim_id": claim_id,
                "prompt_length": prompt_length,
                "start_time": start_wall_s,
                "llm_timeout_s": float(effective_timeout_s),
                "agent_timeout_s": float(agent_timeout_s),
            },
        )

        try:
            completion = await self._llm_service.generate(
                prompt=prompt,
                context=None,
                generation_kwargs=gen_kw,
                claim_id=claim_id,
                timeout_s=effective_timeout_s,
                max_attempts=1,
            )
        except Exception as exc:
            if isinstance(exc, LLMProviderError) and exc.status_code == 404:
                logger.warning(
                    "Model not available, falling back to safe decision",
                    extra={"claim_id": claim_id, "provider": exc.provider},
                )
            is_timeout = type(exc).__name__ == "TimeoutError" or "timeout" in str(exc).lower()

            logger.exception(
                "fraud_llm_failed",
                extra={"claim_id": claim_id, "error": f"{type(exc).__name__}: {exc}"},
            )
            return {
                "fraud_score": 0.5,
                "decision": "INVESTIGATE",
                "confidence": 0.5,
                "entities": {},
                "explanation": {
                    "summary": "AI analysis delayed or unavailable; routed for human review",
                    "key_factors": [
                        f"LLM or transport error: {type(exc).__name__}.",
                        "No structured fraud assessment was produced.",
                    ],
                    "similar_case_reference": "",
                },
                "_llm_failed": True,
                "_timeout_triggered": bool(is_timeout),
                "_llm_timeout": bool(is_timeout),
                "_llm_parse_error": False,
                "_llm_latency_ms": 0,
                "_llm_response_time_ms": 0,
                "_agent_total_time_ms": int((time.perf_counter() - start_perf) * 1000),
            }

        last_raw = completion.text or ""
        total_latency_ms += int(completion.latency_ms)
        llm_response_time_ms = int((time.perf_counter() - start_perf) * 1000)

        logger.info(
            "fraud_llm_completed",
            extra={
                "claim_id": claim_id,
                "llm_time_ms": int(completion.latency_ms),
                "llm_response_time_ms": int(llm_response_time_ms),
                "provider": completion.provider,
                "model": completion.model,
                "timeout_s": float(effective_timeout_s),
            },
        )

        raw_preview, raw_trunc = _truncate_for_log(last_raw)
        logger.info(
            "fraud_llm_raw_output",
            extra={
                "claim_id": claim_id,
                "raw_char_len": len(last_raw),
                "raw_truncated": raw_trunc,
                "raw_text": raw_preview,
            },
        )

        # Requested: explicit raw response message for quick log scanning.
        logger.info("LLM RAW RESPONSE: %s", raw_preview)

        prepared = self._prepare_model_text(last_raw)
        result = self._parse_fraud_response(prepared)
        if not result.ok and int(getattr(settings, "max_llm_retries", 0) or 0) > 0:
            logger.info(
                "fraud_llm_retry_parse_fixup",
                extra={"claim_id": claim_id, "previous_error": result.error},
            )
            try:
                fix_prompt = self._build_fixup_prompt(bad_output=prepared)
                fix_completion = await self._llm_service.generate(
                    prompt=fix_prompt,
                    context=None,
                    generation_kwargs={**gen_kw, "temperature": 0.0},
                    claim_id=claim_id,
                    timeout_s=effective_timeout_s,
                    max_attempts=1,
                )
                last_raw = fix_completion.text or ""
                total_latency_ms += int(fix_completion.latency_ms)
                prepared = self._prepare_model_text(last_raw)
                result = self._parse_fraud_response(prepared)
            except Exception as exc:
                logger.exception(
                    "fraud_llm_retry_failed",
                    extra={"claim_id": claim_id, "error": f"{type(exc).__name__}: {exc}"},
                )
        if result.ok:
            agent_total_ms = int((time.perf_counter() - start_perf) * 1000)
            logger.info(
                "fraud_llm_success",
                extra={
                    "claim_id": claim_id,
                    "provider": completion.provider,
                    "model": completion.model,
                    "llm_time_ms": int(completion.latency_ms),
                    "start_time": start_wall_s,
                    "llm_response_time_ms": int(llm_response_time_ms or 0),
                    "agent_total_time_ms": int(agent_total_ms),
                },
            )
            out: dict[str, Any] = {
                "fraud_score": result.fraud_score,
                "decision": result.decision,
                "confidence": result.confidence,
                "entities": result.entities,
                "explanation": result.explanation,
                "_llm_latency_ms": total_latency_ms,
                "_llm_timeout": False,
                "_llm_parse_error": False,
                "_llm_response_time_ms": int(llm_response_time_ms or 0),
                "_agent_total_time_ms": int(agent_total_ms),
            }
            return out

        logger.warning(
            "fraud_llm_json_parse_failed",
            extra={
                "claim_id": claim_id,
                "error": result.error,
                "start_time": start_wall_s,
                "llm_response_time_ms": int(llm_response_time_ms or 0),
                "agent_total_time_ms": int((time.perf_counter() - start_perf) * 1000),
            },
        )

        return {
            "fraud_score": 0.5,
            "decision": "INVESTIGATE",
            "confidence": 0.5,
            "entities": {},
            "explanation": {
                "summary": "AI analysis delayed or unavailable; routed for human review",
                "key_factors": [
                    "Model response was not usable structured JSON.",
                    "Routed to human review to maintain responsiveness.",
                ],
                "similar_case_reference": "",
            },
            "_llm_failed": True,
            "_timeout_triggered": False,
            "_llm_timeout": False,
            "_llm_parse_error": True,
            "_llm_latency_ms": total_latency_ms,
            "_llm_response_time_ms": int(llm_response_time_ms or 0),
            "_agent_total_time_ms": int((time.perf_counter() - start_perf) * 1000),
        }

    def _prepare_model_text(self, text: str) -> str:
        t = _strip_markdown_json_fence(text)
        return t.strip()

    def _extract_json_candidate(self, text: str) -> Optional[str]:
        balanced = _extract_json_balanced(text)
        if balanced:
            return balanced
        return _extract_json_object_loose(text)

    def _loads_json_with_repair(self, candidate: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
            return None, "json_root_not_object"
        except json.JSONDecodeError as first_exc:
            repaired = _repair_json_candidate(candidate)
            if repaired != candidate:
                try:
                    obj = json.loads(repaired)
                    if isinstance(obj, dict):
                        return obj, None
                    return None, "json_root_not_object_after_repair"
                except json.JSONDecodeError as exc:
                    return None, f"json_decode_error_after_repair: {exc}"
            return None, f"json_decode_error: {first_exc}"

    def _parse_fraud_response(self, prepared_text: str) -> FraudParseResult:
        candidate = self._extract_json_candidate(prepared_text)
        if not candidate:
            return FraudParseResult(
                0.5,
                "INVESTIGATE",
                0.5,
                _default_explanation(),
                {},
                False,
                "no_json_object_extracted",
            )

        payload, err = self._loads_json_with_repair(candidate)
        if payload is None:
            return FraudParseResult(
                0.5,
                "INVESTIGATE",
                0.5,
                _default_explanation(),
                {},
                False,
                err or "json_load_failed",
            )

        if "fraud_score" not in payload:
            return FraudParseResult(
                0.5,
                "INVESTIGATE",
                0.5,
                _default_explanation(),
                {},
                False,
                "missing_fraud_score",
            )

        fraud_score_raw: Optional[Any] = payload.get("fraud_score")
        decision_raw: Optional[Any] = payload.get("decision")
        reasons_raw: Optional[Any] = payload.get("reasons")
        confidence_raw: Optional[Any] = payload.get("confidence")
        explanation_raw: Optional[Any] = payload.get("explanation")
        entities_raw: Optional[Any] = payload.get("entities")

        fraud_score = 0.0
        try:
            if isinstance(fraud_score_raw, (int, float, str)):
                fraud_score = float(fraud_score_raw)
        except (TypeError, ValueError):
            fraud_score = 0.0
        fraud_score = min(1.0, max(0.0, fraud_score))

        decision = "INVESTIGATE"
        if isinstance(decision_raw, str) and decision_raw.strip():
            d_u = decision_raw.strip().upper()
            # Normalize synonyms to internal canonical words used across the pipeline.
            if d_u in ("APPROVE", "APPROVED"):
                d_u = "APPROVED"
            elif d_u in ("REJECT", "REJECTED"):
                d_u = "REJECTED"
            if d_u in ("APPROVED", "REJECTED", "INVESTIGATE"):
                decision = d_u

        confidence = 0.5
        try:
            if isinstance(confidence_raw, (int, float, str)):
                confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))

        # Prefer strict "reasons" output (requested). Map it into our existing explanation shape.
        reasons: list[str] = []
        if isinstance(reasons_raw, list):
            reasons = [str(x).strip() for x in reasons_raw if str(x).strip()][:8]
        if reasons:
            factors = (reasons + ["Insufficient structured explanation from model."])[:2]
            explanation = {
                "summary": "Fraud assessment produced by LLM.",
                "key_factors": factors,
                "similar_case_reference": "",
            }
        else:
            explanation = _normalize_explanation(explanation_raw, payload)

        entities: dict[str, Any] = {}
        if isinstance(entities_raw, dict):
            entities = {str(k): v for k, v in entities_raw.items()}

        factors_list = explanation.get("key_factors")
        ok = bool(str(explanation.get("summary") or "").strip() and isinstance(factors_list, list))
        if not ok:
            return FraudParseResult(
                fraud_score,
                decision,
                confidence,
                explanation,
                entities,
                False,
                "schema_incomplete",
            )

        return FraudParseResult(
            fraud_score,
            decision,
            confidence,
            explanation,
            entities,
            True,
            None,
        )

    def _parse_fraud_json(self, text: str) -> Tuple[float, str, float, dict[str, Any], dict[str, Any], bool]:
        """Back-compat tuple parse (used internally; same semantics as :meth:`_parse_fraud_response`)."""
        r = self._parse_fraud_response(self._prepare_model_text(text))
        return r.fraud_score, r.decision, r.confidence, r.explanation, r.entities, r.ok


def _normalize_explanation(explanation_raw: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(explanation_raw, dict):
        summary = str(explanation_raw.get("summary") or "").strip()
        kf = explanation_raw.get("key_factors")
        factors: list[str] = []
        if isinstance(kf, list):
            factors = [str(x).strip() for x in kf if str(x).strip()][:8]
        ref = str(explanation_raw.get("similar_case_reference") or "").strip()
        if not summary:
            summary = "No summary provided."
        if len(factors) < 2:
            legacy = payload.get("explanation")
            if isinstance(legacy, str) and legacy.strip():
                factors = [legacy.strip()]
            if len(factors) < 2:
                factors = (factors + ["Insufficient structured explanation from model."])[:4]
        # Enforce "max 2 points" contract while keeping the parser requirement (>=2).
        factors = (factors + ["Insufficient structured explanation from model."])[:2]
        return {"summary": summary, "key_factors": factors[:2], "similar_case_reference": ref}

    legacy = payload.get("explanation")
    if isinstance(legacy, str) and legacy.strip():
        return {
            "summary": legacy.strip()[:300],
            "key_factors": [legacy.strip()[:500]],
            "similar_case_reference": "",
        }
    return _default_explanation()


__all__ = ["FraudAgent", "FraudAgentInput"]
