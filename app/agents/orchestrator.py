from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from typing import Any

from fastapi import Depends

from app.agents.decision_agent import DecisionAgent, image_severity_to_score
from app.agents.fraud_agent import FraudAgent
from app.agents.policy_agent import PolicyAgent
from app.core.config import settings
from app.core.dependencies import get_embedding_service, get_hitl_service, get_llm_service, get_vector_store
from app.models.schemas import DecisionMetadata, InferenceRequest, InferenceResponse
from app.services.context_builder import ContextBuilder
from app.services.dl_fraud_model import DeepFraudModel
from app.services.embedding_service import EmbeddingService
from app.services.hitl_service import HitlDecision, HitlService
from app.services.image_cnn_service import ImageCNNService
from app.services.image_service import ImageService
from app.services.llm_service import LLMService
from app.services.metrics import metrics
from app.services.reranker import LightweightReranker
from app.services.retriever import ClaimRetriever, RetrievalParams
from app.services.vector_store import (
    SimilarHit,
    VectorStore,
    compute_calibrated_confidence,
    majority_review_from_similar_hits,
)

logger = logging.getLogger(__name__)


def _explanation_storage_value(explanation: Any) -> str:
    if isinstance(explanation, dict):
        return json.dumps(explanation, ensure_ascii=False)
    text = str(explanation or "").strip()
    if not text:
        return json.dumps(
            {"summary": "No explanation.", "key_factors": ["Missing explanation."], "similar_case_reference": ""},
            ensure_ascii=False,
        )
    return text


def _jpeg_thumbnail_base64(
    image_bytes: bytes,
    *,
    max_side: int = 360,
    quality: int = 72,
    max_b64_chars: int = 96_000,
) -> str:
    """Small JPEG preview for UI (Chroma metadata); empty string on failure."""
    try:
        from io import BytesIO

        from PIL import Image

        im = Image.open(BytesIO(image_bytes)).convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        out = BytesIO()
        im.save(out, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(out.getvalue()).decode("ascii")
        if len(b64) <= max_b64_chars:
            return b64
        im2 = im.copy()
        im2.thumbnail((220, 220), Image.Resampling.LANCZOS)
        out2 = BytesIO()
        im2.save(out2, format="JPEG", quality=62, optimize=True)
        return base64.b64encode(out2.getvalue()).decode("ascii")[:max_b64_chars]
    except Exception:
        logger.exception("image_preview_encode_failed")
        return ""


def _strip_image_transport_fields(claim: dict[str, Any]) -> tuple[dict[str, Any], bytes | None]:
    """Remove image transport keys from the working claim dict and return raw image bytes if any."""
    c = dict(claim)
    raw = c.pop("_image_bytes", None)
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if raw is not None and not isinstance(raw, bytes):
        raw = None
    b64 = c.pop("image_base64", None)
    if raw is None and b64 is not None:
        s = str(b64).strip()
        if s.startswith("data:") and "," in s:
            s = s.split(",", 1)[1].strip()
        if s:
            try:
                raw = base64.b64decode(s, validate=True)
            except (binascii.Error, ValueError):
                raw = None
    return c, raw


class InsurFlowOrchestrator:
    """Micro-insurance claim pipeline: parallel fraud + policy checks, then a decision."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        hitl_service: HitlService,
    ) -> None:
        self._llm_service = llm_service
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._hitl_service = hitl_service
        self._fraud_agent = FraudAgent(
            llm_service=llm_service,
        )
        self._policy_agent = PolicyAgent()
        self._decision_agent = DecisionAgent()
        self._dl_fraud_model = DeepFraudModel(enabled=settings.dl_fraud_enabled)
        self._rag_enabled = settings.rag_enabled
        # Aggressive latency optimization: keep RAG very small for FraudAgent.
        self._rag_top_k = 1
        self._retriever = ClaimRetriever(vector_store)
        self._context_builder = ContextBuilder(max_tokens=settings.rag_context_max_tokens)
        self._reranker = LightweightReranker() if settings.rag_rerank_enabled else None
        self._image_service = ImageService()
        self._image_cnn_service = ImageCNNService(fallback_service=self._image_service)

    async def _dl_fraud_score_safe(
        self,
        *,
        claim_amount: float,
        structured: dict[str, Any],
        embedding: list[float] | None,
    ) -> float | None:
        if not settings.dl_fraud_enabled:
            return None
        try:
            return await self._dl_fraud_model.predict_async(
                claim_amount=claim_amount,
                structured=structured,
                embedding=embedding,
            )
        except Exception as exc:
            logger.exception(
                "dl_fraud_inference_failed",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
            return None

    async def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        provider_override: str | None = None
        if request.task:
            task_l = request.task.strip().lower()
            if task_l == "cheap":
                provider_override = "ollama"
            elif task_l == "complex":
                provider_override = "openai"

        completion = await self._llm_service.generate(
            prompt=request.prompt,
            context=request.context,
            model=request.model,
            provider=provider_override,
        )
        return InferenceResponse(
            text=completion.text,
            provider=completion.provider,
            model=completion.model,
            tokens=completion.tokens,
            cost=completion.cost,
            latency=completion.latency_ms,
            confidence=completion.confidence,
        )

    async def process_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Embed → parallel (RAG + DL + image) → parallel (fraud LLM + policy) → decision → memory upsert."""
        workflow_start = time.perf_counter()
        claim_work, image_bytes = _strip_image_transport_fields(claim)
        logger.info(
            "orchestrator_claim_start",
            extra={
                "claim_keys": list(claim_work.keys()),
                "has_image": bool(image_bytes),
                "image_bytes_len": len(image_bytes) if image_bytes else 0,
            },
        )

        claim_id = str(claim_work.get("claim_id") or "").strip() or "unknown"
        claim_description = str(claim_work.get("description") or "").strip()
        if not claim_description:
            claim_description = json.dumps(claim_work, ensure_ascii=False, default=str)

        similar_context = ""
        similar_hits: list[SimilarHit] = []
        embedding_for_store: list[float] | None = None
        embedding_status = "fail"
        rag_context_used = False
        embedding_ms = 0.0
        retrieval_ms = 0.0

        try:
            t_emb = time.perf_counter()
            embedding_for_store = await self._embedding_service.embed(claim_description)
            embedding_ms = (time.perf_counter() - t_emb) * 1000
            if embedding_for_store:
                embedding_status = "success"
            else:
                embedding_status = "fail"
        except Exception as exc:
            embedding_status = "fail"
            embedding_for_store = None
            logger.exception(
                "embedding_failed",
                extra={"claim_id": claim_id, "error": f"{type(exc).__name__}: {exc}"},
            )

        def _run_rag_sync() -> tuple[list[SimilarHit], str, float, bool]:
            if not self._rag_enabled or not embedding_for_store:
                return [], "", 0.0, False
            t_ret = time.perf_counter()
            try:
                meta_equal = claim_work.get("rag_metadata_filter")
                if not isinstance(meta_equal, dict):
                    meta_equal = None
                decision_filter = str(claim_work.get("rag_filter_decision") or "").strip() or None
                product_code = str(claim_work.get("product_code") or "").strip() or None
                hits = self._retriever.retrieve(
                    RetrievalParams(
                        claim_description=claim_description,
                        query_embedding=embedding_for_store,
                        exclude_claim_id=claim_id,
                        top_k=self._rag_top_k,
                        decision_equal=decision_filter,
                        metadata_equal=meta_equal,
                        product_code_equal=product_code,
                    )
                )
                if self._reranker is not None:
                    hits = self._reranker.rerank(hits, claim=claim_work, product_code=product_code)
                similar_hits_local = hits[: self._rag_top_k]
                similar_context_local = self._context_builder.build(similar_hits_local)
                rms = (time.perf_counter() - t_ret) * 1000
                used = bool(similar_context_local.strip())
                if similar_context_local:
                    logger.info("similar_claims_context_ready", extra={"claim_id": claim_id})
                return similar_hits_local, similar_context_local, rms, used
            except Exception as exc:
                logger.exception(
                    "retrieval_context_failed",
                    extra={"claim_id": claim_id, "error": f"{type(exc).__name__}: {exc}"},
                )
                return [], "", (time.perf_counter() - t_ret) * 1000, False

        async def _rag_async() -> tuple[list[SimilarHit], str, float, bool]:
            return await asyncio.to_thread(_run_rag_sync)

        structured_for_dl = {
            "policy_limit": claim_work.get("policy_limit"),
            "product_code": claim_work.get("product_code"),
            "currency": claim_work.get("currency"),
            "description": claim_work.get("description"),
            "incident_date": claim_work.get("incident_date"),
            "policyholder_id": claim_work.get("policyholder_id"),
        }
        try:
            dl_amount = float(claim_work.get("claim_amount") or 0.0)
        except (TypeError, ValueError):
            dl_amount = 0.0

        # Decide early whether to use RAG context for the fraud prompt.
        # Goal: keep prompts tiny for simple/low-risk cases to reduce latency/timeouts.
        # Note: We may still retrieve for other uses later, but this prevents sending verbose
        # similar-claims text into the FraudAgent unless it adds value.
        def _fraud_case_is_complex(*, amount: float, has_image: bool) -> bool:
            desc_l_local = str(claim_description or "").lower()
            text_claims_damage_local = "crack" in desc_l_local
            # No image or unknown signals => treat as more complex (needs language reasoning).
            if not has_image:
                return True
            # Higher payout potential => more complex.
            if amount >= 500:
                return True
            # Damage-described claims are higher leverage for fraud consistency.
            if text_claims_damage_local:
                return True
            return False

        async def _dl_job() -> tuple[float | None, float]:
            t0 = time.perf_counter()
            prob = await self._dl_fraud_score_safe(
                claim_amount=dl_amount,
                structured=structured_for_dl,
                embedding=embedding_for_store,
            )
            return prob, (time.perf_counter() - t0) * 1000

        async def _image_job() -> tuple[dict[str, Any] | None, float]:
            if not settings.enable_image_analysis or not image_bytes:
                return None, 0.0
            t0 = time.perf_counter()
            try:
                feats = await asyncio.to_thread(self._image_cnn_service.analyze, image_bytes)
                sev_raw = str((feats or {}).get("severity") or "").strip().lower()
                # If the CNN result is low-confidence / unknown, fall back to heuristic
                # while preserving CNN confidence for observability and explainability.
                if sev_raw == "unknown":
                    heur = await asyncio.to_thread(self._image_service.analyze, image_bytes)
                    feats = {
                        **(heur if isinstance(heur, dict) else {}),
                        "source": "cnn_low_confidence_fallback"
                        if str((feats or {}).get("source") or "").strip().lower() == "cnn_low_confidence"
                        else "cnn_unknown_severity_fallback",
                        "signals": {
                            **(
                                heur.get("signals")  # type: ignore[union-attr]
                                if isinstance(heur, dict) and isinstance(heur.get("signals"), dict)
                                else {}
                            ),
                            **(
                                feats.get("signals")  # type: ignore[union-attr]
                                if isinstance(feats, dict) and isinstance(feats.get("signals"), dict)
                                else {}
                            ),
                            "cnn_used": False,
                        },
                    }
                ms = (time.perf_counter() - t0) * 1000
                sigs_for_log = (feats or {}).get("signals") if isinstance(feats, dict) else None
                if isinstance(sigs_for_log, dict) and sigs_for_log.get("cnn_label") is not None:
                    cnn_label = str(sigs_for_log.get("cnn_label") or "")
                else:
                    cnn_label = str((feats or {}).get("label") or "")
                if isinstance(sigs_for_log, dict) and sigs_for_log.get("cnn_confidence") is not None:
                    try:
                        cnn_conf = float(sigs_for_log.get("cnn_confidence") or 0.0)
                    except (TypeError, ValueError):
                        cnn_conf = 0.0
                else:
                    try:
                        cnn_conf = float((feats or {}).get("confidence") or 0.0)
                    except (TypeError, ValueError):
                        cnn_conf = 0.0
                cnn_used = True
                try:
                    sigs = (feats or {}).get("signals")
                    if isinstance(sigs, dict) and "cnn_used" in sigs:
                        cnn_used = bool(sigs.get("cnn_used"))
                except Exception:
                    cnn_used = True
                logger.info(
                    "image_analysis_complete",
                    extra={
                        "claim_id": claim_id,
                        "image_processing_time_ms": round(ms, 2),
                        "cnn_label": cnn_label,
                        "cnn_confidence": round(min(1.0, max(0.0, cnn_conf)), 4),
                        "cnn_used": cnn_used,
                        "cnn_latency": round(ms, 2),
                        "image_features": feats,
                    },
                )
                return feats, ms
            except Exception as exc:
                ms = (time.perf_counter() - t0) * 1000
                logger.exception(
                    "image_analysis_failed",
                    extra={"claim_id": claim_id, "error": f"{type(exc).__name__}: {exc}"},
                )
                return None, ms

        if settings.enable_parallel_execution:
            rag_pack, dl_pack, img_pack = await asyncio.gather(
                _rag_async(),
                _dl_job(),
                _image_job(),
            )
        else:
            rag_pack = await _rag_async()
            dl_pack = await _dl_job()
            img_pack = await _image_job()

        similar_hits, similar_context, retrieval_ms, rag_context_used = rag_pack
        dl_fraud_probability, dl_ms = dl_pack
        image_feats_raw, image_processing_ms = img_pack

        retrieval_count = len(similar_hits)

        image_damage_type_store = ""
        image_severity_store = ""
        image_severity_score_for_fusion: float | None = None
        image_features: dict[str, Any]
        if image_feats_raw:
            image_features = {"present": True, **image_feats_raw}
            sev_s = image_severity_to_score(image_feats_raw.get("severity"))
            try:
                ic = float(image_feats_raw.get("confidence") or 0.0)
            except (TypeError, ValueError):
                ic = 0.0
            ic = min(1.0, max(0.0, ic))
            image_severity_score_for_fusion = min(1.0, max(0.0, sev_s * (0.35 + 0.65 * ic)))
            image_damage_type_store = str(image_feats_raw.get("damage_type") or "")
            image_severity_store = str(image_feats_raw.get("severity") or "")
        else:
            image_features = {
                "present": False,
                "damage_type": "n/a",
                "severity": "n/a",
                "confidence": 0.0,
            }
            if not settings.enable_image_analysis:
                image_features["reason"] = "disabled"
            elif not image_bytes:
                image_features["reason"] = "no_image"
            else:
                image_features["reason"] = "unavailable"

        # Cross-modal fraud consistency check (description vs image label).
        # Rule: if description claims damage ("crack") but image shows no evidence, flag and override to INVESTIGATE.
        desc_l = str(claim_description or "").lower()
        text_claims_damage = "crack" in desc_l
        cnn_label_raw = ""
        try:
            sigs = image_features.get("signals") if isinstance(image_features, dict) else None
            if isinstance(sigs, dict) and sigs.get("cnn_label") is not None:
                cnn_label_raw = str(sigs.get("cnn_label") or "").strip()
        except Exception:
            cnn_label_raw = ""
        if not cnn_label_raw:
            cnn_label_raw = str(image_features.get("label") or image_features.get("damage_type") or "").strip()
        cnn_label = cnn_label_raw.lower()
        # Conservative: treat unknown/uncertain as "no evidence" to avoid approving potential fraud.
        image_shows_damage = cnn_label in ("minor_crack", "major_crack")
        if cnn_label in ("unknown", "uncertain"):
            image_shows_damage = False
        fraud_signal: str | None = None
        if bool(image_features.get("present")) and text_claims_damage and not image_shows_damage:
            fraud_signal = "image_text_mismatch"
            logger.info(
                "fraud_consistency_mismatch",
                extra={
                    "claim_id": claim_id,
                    "fraud_signal": fraud_signal,
                    "text_claims_damage": text_claims_damage,
                    "cnn_label": cnn_label_raw,
                    "image_shows_damage": image_shows_damage,
                },
            )

        fraud_input = dict(claim_work)
        # Only include similar-claims context for complex cases to keep prompt payload small.
        # FraudAgent will still ignore it unless include_similar_claims_context is true.
        try:
            claim_amount = float(claim_work.get("claim_amount") or 0.0)
        except (TypeError, ValueError):
            claim_amount = 0.0
        has_image = bool(image_features.get("present"))
        fraud_complex = _fraud_case_is_complex(amount=float(claim_amount), has_image=has_image) or bool(fraud_signal)
        fraud_input["similar_claims_context"] = similar_context if fraud_complex else ""
        fraud_input["include_similar_claims_context"] = bool(fraud_complex)
        fraud_input["image_features"] = image_features
        fraud_input["text_claims_damage"] = bool(text_claims_damage)
        fraud_input["image_text_mismatch"] = bool(fraud_signal == "image_text_mismatch")
        # Conditional LLM execution: skip for obvious cases to hit latency SLA.
        image_severity = str(image_features.get("severity") or "").strip().lower()
        try:
            policy_limit = float(claim_work.get("policy_limit") or 0.0)
        except (TypeError, ValueError):
            policy_limit = 0.0
        fraud_input["amount_over_policy_limit"] = bool(policy_limit > 0 and claim_amount > policy_limit)

        async def _fraud_run() -> dict[str, Any]:
            if image_severity == "low" and claim_amount < 500:
                logger.info(
                    "fraud_llm_skipped_rule_approved",
                    extra={
                        "claim_id": claim_id,
                        "image_severity": image_severity,
                        "claim_amount": claim_amount,
                    },
                )
                # Return an APPROVED fraud result directly (no LLM) while preserving expected shape.
                return {
                    "fraud_score": 0.05,
                    "decision": "APPROVED",
                    "confidence": 0.85,
                    "entities": {},
                    "explanation": {
                        "summary": "Low-severity damage and small claim amount — LLM skipped for latency.",
                        "key_factors": [
                            "Image severity is low.",
                            "Claim amount is below 500.",
                        ],
                        "similar_case_reference": "",
                    },
                    "_llm_skipped": True,
                    "_llm_latency_ms": 0,
                    "_timeout_triggered": False,
                }

            if image_severity == "high" and policy_limit > 0 and claim_amount > policy_limit:
                logger.info(
                    "fraud_llm_skipped_rule_rejected",
                    extra={
                        "claim_id": claim_id,
                        "image_severity": image_severity,
                        "claim_amount": claim_amount,
                        "policy_limit": policy_limit,
                    },
                )
                return {
                    "fraud_score": 0.95,
                    "decision": "REJECTED",
                    "confidence": 0.9,
                    "entities": {},
                    "explanation": {
                        "summary": "High-severity damage and claim exceeds policy limit — LLM skipped for latency.",
                        "key_factors": [
                            "Image severity is high.",
                            "Claim amount exceeds policy limit.",
                        ],
                        "similar_case_reference": "",
                    },
                    "_llm_skipped": True,
                    "_llm_latency_ms": 0,
                    "_timeout_triggered": False,
                }

            # Best practice: do NOT wrap the whole agent in a timeout.
            # Only the LLM call itself should be time-bounded (handled inside FraudAgent/LLM router).
            # This prevents false timeouts after the LLM already responded (e.g., during parsing).
            return await self._fraud_agent.run(fraud_input)

        async def _policy_run() -> dict[str, Any]:
            return await self._policy_agent.run(claim_work)

        async def _fraud_and_policy() -> tuple[dict[str, Any], dict[str, Any]]:
            if settings.enable_parallel_execution:
                return await asyncio.gather(_fraud_run(), _policy_run())
            fraud_o = await _fraud_run()
            policy_o = await _policy_run()
            return fraud_o, policy_o

        try:
            fraud_out, policy_out = await asyncio.wait_for(
                _fraud_and_policy(),
                timeout=float(settings.claim_timeout_s),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "claim_pipeline_timeout",
                extra={"claim_id": claim_id, "timeout_s": float(settings.claim_timeout_s)},
            )
            fraud_out = {
                "fraud_score": 0.5,
                "decision": "INVESTIGATE",
                "confidence": 0.5,
                "entities": {},
                "explanation": {
                    "summary": "AI analysis delayed or unavailable; routed for human review",
                    "key_factors": [
                        f"Pipeline timed out after {float(settings.claim_timeout_s):.0f}s.",
                        "Returned a conservative fallback decision to keep the system responsive.",
                    ],
                    "similar_case_reference": "",
                },
                "_llm_failed": True,
                "_timeout_triggered": True,
                "_llm_latency_ms": int(float(settings.claim_timeout_s) * 1000),
            }
            policy_out = {"policy_valid": True, "reason": "policy_check_skipped_due_to_timeout"}

        llm_ms = float(fraud_out.get("_llm_latency_ms") or 0.0)
        # Timeout is only true when we actually hit a wall-clock cap (not when parsing fails).
        timeout_triggered = bool(fraud_out.get("_timeout_triggered"))
        llm_skipped = bool(fraud_out.get("_llm_skipped"))
        llm_used = not llm_skipped
        if llm_skipped:
            decision_source = "rule"
        elif bool(fraud_out.get("_llm_failed")) or timeout_triggered:
            decision_source = "fallback"
        else:
            decision_source = "llm"

        if dl_fraud_probability is not None:
            logger.info(
                "dl_fraud_score",
                extra={
                    "claim_id": claim_id,
                    "dl_fraud_probability": dl_fraud_probability,
                    "dl_backend": self._dl_fraud_model.backend,
                },
            )

        fraud_llm_failed = bool(fraud_out.get("_llm_failed"))

        if fraud_llm_failed:
            def _clamp01(x: Any, default: float = 0.0) -> float:
                try:
                    f = float(x)
                except (TypeError, ValueError):
                    f = float(default)
                return min(1.0, max(0.0, f))

            def _amount_risk_score(*, claim_amount_value: float, policy_limit_value: float) -> float:
                """Deterministic payout-risk proxy in [0,1]."""
                try:
                    amt = float(claim_amount_value or 0.0)
                except (TypeError, ValueError):
                    amt = 0.0
                try:
                    lim = float(policy_limit_value or 0.0)
                except (TypeError, ValueError):
                    lim = 0.0
                amt = max(0.0, amt)
                lim = max(0.0, lim)

                if lim > 0:
                    # Ratio is interpretable and stable.
                    return _clamp01(amt / lim, default=0.0)

                # No limit available: approximate risk from absolute amount.
                # Below 500 is generally low; above 2000 is high in this demo domain.
                if amt <= 0:
                    return 0.0
                if amt < 500:
                    return 0.2
                if amt < 2000:
                    # Linearly ramp from 0.35 to 0.85 across [500, 2000)
                    return _clamp01(0.35 + (amt - 500.0) * (0.5 / 1500.0), default=0.5)
                return 0.95

            def _fallback_heuristic_decision(
                *,
                image_severity_label: str,
                claim_amount_value: float,
                policy_limit_value: float,
                image_score_value: float | None,
            ) -> dict[str, Any]:
                sev = (image_severity_label or "").strip().lower()
                try:
                    amt = float(claim_amount_value or 0.0)
                except (TypeError, ValueError):
                    amt = 0.0
                try:
                    lim = float(policy_limit_value or 0.0)
                except (TypeError, ValueError):
                    lim = 0.0
                amt = max(0.0, amt)
                lim = max(0.0, lim)

                image_score = None if image_score_value is None else _clamp01(image_score_value, default=0.0)
                amount_risk = _amount_risk_score(claim_amount_value=amt, policy_limit_value=lim)
                # Risk score blending (used only when LLM fails).
                final_score = (
                    (image_score * 0.6) + (amount_risk * 0.4)
                    if image_score is not None
                    else amount_risk
                )
                final_score = _clamp01(final_score, default=0.5)

                # Fallback decision logic (deterministic, explainable).
                if sev == "low" and lim > 0 and amt <= lim:
                    return {
                        "decision": "APPROVED",
                        "confidence_score": 0.7,
                        "reason": "LLM unavailable; low severity and within policy limit",
                        "image_score": image_score,
                        "amount_risk": round(amount_risk, 4),
                        "final_score": round(final_score, 4),
                    }
                if sev == "low" and lim > 0 and amt > lim:
                    return {
                        "decision": "INVESTIGATE",
                        "confidence_score": 0.5,
                        "reason": "LLM unavailable; low severity but claim exceeds policy limit",
                        "image_score": image_score,
                        "amount_risk": round(amount_risk, 4),
                        "final_score": round(final_score, 4),
                    }
                if sev == "medium":
                    return {
                        "decision": "INVESTIGATE",
                        "confidence_score": 0.5,
                        "reason": "LLM unavailable; medium severity requires review",
                        "image_score": image_score,
                        "amount_risk": round(amount_risk, 4),
                        "final_score": round(final_score, 4),
                    }
                if sev == "high":
                    return {
                        "decision": "REJECTED",
                        "confidence_score": 0.7,
                        "reason": "LLM unavailable; high severity indicates likely rejection",
                        "image_score": image_score,
                        "amount_risk": round(amount_risk, 4),
                        "final_score": round(final_score, 4),
                    }

                # Unknown / missing severity: keep safety.
                return {
                    "decision": "INVESTIGATE",
                    "confidence_score": 0.5,
                    "reason": "LLM unavailable; insufficient image severity signal",
                    "image_score": image_score,
                    "amount_risk": round(amount_risk, 4),
                    "final_score": round(final_score, 4),
                }

            # Improve explainability for fraud output + downstream UI when LLM is unavailable.
            # Keep deterministic, safety-preserving heuristics.
            fallback_summary = "LLM unavailable; decision based on policy and risk heuristics"
            fallback_fraud_expl = {
                "summary": "AI analysis delayed or unavailable; routed for human review",
                "key_factors": [
                    "Primary LLM analysis failed or timed out.",
                    f"Fallback inputs: image_severity={image_severity or 'n/a'}, claim_amount={claim_amount:.2f}, policy_limit={policy_limit:.2f}.",
                ],
                "similar_case_reference": "",
            }
            if isinstance(fraud_out.get("explanation"), dict):
                fraud_out["explanation"] = fallback_fraud_expl
            else:
                fraud_out["explanation"] = fallback_fraud_expl

            policy_valid = bool((policy_out or {}).get("policy_valid"))
            if not policy_valid:
                decision_out: dict[str, Any] = {
                    "decision": "REJECTED",
                    "confidence_score": 0.9,
                    "explanation": "Policy check failed; fraud model unavailable.",
                }
            else:
                decision_out = _fallback_heuristic_decision(
                    image_severity_label=image_severity,
                    claim_amount_value=claim_amount,
                    policy_limit_value=policy_limit,
                    image_score_value=image_severity_score_for_fusion,
                )
                boundary_case = bool(policy_limit > 0 and claim_amount == policy_limit)
                logger.info(
                    "fallback_decision_heuristics",
                    extra={
                        "claim_id": claim_id,
                        "image_severity": image_severity or "n/a",
                        "claim_amount": claim_amount,
                        "policy_limit": policy_limit,
                        "boundary_case": boundary_case,
                        "decision": str(decision_out.get("decision") or ""),
                        "confidence_score": float(decision_out.get("confidence_score") or 0.0),
                    },
                )
                # Improve explanation payload (requested contract).
                decision_out["summary"] = fallback_summary
                decision_out["explanation"] = {
                    "summary": fallback_summary,
                    "image_severity": image_severity or "n/a",
                    "claim_amount": round(float(claim_amount), 2),
                    "policy_limit": round(float(policy_limit), 2),
                    "reason": str(decision_out.get("reason") or ""),
                }
        else:
            fraud_for_decision = {
                k: v
                for k, v in fraud_out.items()
                if k not in ("_llm_failed", "_llm_latency_ms")
            }
            decision_in: dict[str, Any] = {
                "fraud": fraud_for_decision,
                "policy": policy_out,
                "similar_majority_review": majority_review_from_similar_hits(similar_hits),
                "fraud_probability_dl": dl_fraud_probability,
                "dl_fusion_llm_weight": settings.dl_fraud_fusion_llm_weight,
                "dl_fusion_dl_weight": settings.dl_fraud_fusion_dl_weight,
                "image_severity_score": image_severity_score_for_fusion,
                "image_fusion_weight": settings.image_fusion_weight,
            }
            decision_out = await self._decision_agent.run(decision_in)
            logger.info(
                "decision_after_fusion",
                extra={
                    "claim_id": claim_id,
                    "decision": decision_out.get("decision"),
                    "fused_fraud_score": decision_out.get("fused_fraud_score"),
                    "fraud_score_llm": decision_out.get("fraud_score_llm"),
                    "fraud_probability_dl": decision_out.get("fraud_probability_dl"),
                    "image_severity_score": decision_out.get("image_severity_score"),
                    "image_fusion_weight": settings.image_fusion_weight,
                },
            )

        # Priority override: description-image mismatch should override rule-based and fallback approvals.
        if fraud_signal == "image_text_mismatch":
            decision_out = {
                **(decision_out or {}),
                "decision": "INVESTIGATE",
                "confidence_score": 0.6,
                "reason": "Claim describes damage but image shows no evidence",
            }
            # Keep explainability: persist the fraud signal in the stored explanation (2 bullet max contract).
            expl = fraud_out.get("explanation")
            if not isinstance(expl, dict):
                expl = {"summary": "", "key_factors": [], "similar_case_reference": ""}
            summary = str(expl.get("summary") or "").strip() or "Fraud signal detected from consistency checks."
            kf_raw = expl.get("key_factors")
            kf: list[str] = []
            if isinstance(kf_raw, list):
                kf = [str(x).strip() for x in kf_raw if str(x).strip()]
            kf = (kf + ["Fraud signal: description-image mismatch"])[:2]
            expl["summary"] = summary
            expl["key_factors"] = kf
            fraud_out["explanation"] = expl

        base_confidence = float(decision_out.get("confidence_score") or 0.0)
        calibrated = compute_calibrated_confidence(
            confidence=base_confidence,
            model_decision=str(decision_out.get("decision") or ""),
            similar_hits=similar_hits,
        )

        hitl = self._hitl_service.evaluate(
            decision=str(decision_out.get("decision") or ""),
            confidence=float(calibrated),
        )
        hitl_decision_reason = hitl.reason or ("HITL required." if hitl.needs_hitl else "Not required.")

        # Keep safety: for borderline fallback cases, force HITL deterministically.
        # "Borderline" means the blended fallback score is near the mid-band or amount is near the limit.
        if str(decision_source).lower() == "fallback":
            final_score_raw = decision_out.get("final_score")
            final_score = None
            try:
                if final_score_raw is not None:
                    final_score = float(final_score_raw)
            except (TypeError, ValueError):
                final_score = None
            near_limit = bool(policy_limit > 0 and claim_amount >= 0.85 * policy_limit)
            borderline = bool(final_score is not None and 0.45 <= final_score <= 0.6) or near_limit
            if borderline and not hitl.needs_hitl:
                hitl = HitlDecision(
                    needs_hitl=True,
                    reason="Borderline fallback case (LLM unavailable) routed for human review.",
                )
            # Fallback awareness (requested): allow clearly low-risk fallback approvals to bypass HITL.
            # Otherwise, keep safety and route fallback decisions to human review.
            fb_dec = str(decision_out.get("decision") or "").strip().upper()
            if str(image_severity or "").strip().lower() == "low" and float(claim_amount) <= float(policy_limit or 0.0):
                if fb_dec == "APPROVED":
                    hitl = HitlDecision(needs_hitl=False, reason="Not required (low-risk fallback approval).")
            else:
                # Any non-low / over-limit fallback outcome should be reviewed.
                if not hitl.needs_hitl:
                    hitl = HitlDecision(needs_hitl=True, reason="Fallback used; requires review due to risk signals.")

        hitl_decision_reason = hitl.reason or ("HITL required." if hitl.needs_hitl else "Not required.")
        review_reason = hitl_decision_reason

        logger.info(
            "hitl_decision_reason",
            extra={
                "claim_id": claim_id,
                "decision": str(decision_out.get("decision") or ""),
                "decision_source": decision_source,
                "calibrated_confidence": float(calibrated),
                "hitl_needed": hitl.needs_hitl,
                "hitl_decision_reason": hitl_decision_reason,
            },
        )

        fraud_for_client = {
            k: v for k, v in fraud_out.items() if k not in ("_llm_failed", "_llm_latency_ms")
        }

        # ---- Attribution metadata for UI transparency ----
        img_signals = None
        try:
            sigs = image_features.get("signals") if isinstance(image_features, dict) else None
            if isinstance(sigs, dict):
                img_signals = sigs
        except Exception:
            img_signals = None

        def _clamp01(x: Any, default: float = 0.0) -> float:
            try:
                f = float(x)
            except (TypeError, ValueError):
                f = float(default)
            return min(1.0, max(0.0, f))

        # CNN fields required by UI contract.
        cnn_label_raw = ""
        if isinstance(img_signals, dict) and img_signals.get("cnn_label") is not None:
            cnn_label_raw = str(img_signals.get("cnn_label") or "").strip()
        if not cnn_label_raw:
            cnn_label_raw = str(image_features.get("label") or image_features.get("damage_type") or "").strip()
        cnn_label = (cnn_label_raw or "unknown").strip().lower()

        cnn_confidence = 0.0
        if isinstance(img_signals, dict) and img_signals.get("cnn_confidence") is not None:
            cnn_confidence = _clamp01(img_signals.get("cnn_confidence"), default=0.0)
        else:
            cnn_confidence = _clamp01(image_features.get("confidence"), default=0.0)

        cnn_severity_raw = ""
        if isinstance(img_signals, dict) and img_signals.get("cnn_severity") is not None:
            cnn_severity_raw = str(img_signals.get("cnn_severity") or "").strip().lower()
        if not cnn_severity_raw:
            cnn_severity_raw = str(image_features.get("severity") or "").strip().lower()
        cnn_severity = cnn_severity_raw if cnn_severity_raw in ("low", "medium", "high") else "unknown"

        # Orchestrator rule: CNN is "used" iff it produced a non-unknown label.
        cnn_used = bool(bool(image_features.get("present")) and settings.enable_image_analysis and cnn_label != "unknown")

        rules_used = True  # policy + deterministic logic always contribute to final decision.

        llm_status: str = "skipped" if llm_skipped else ("failed" if decision_source == "fallback" else "used")
        llm_failure_reason: str | None = None
        if llm_status == "failed":
            llm_failure_reason = "timeout" if timeout_triggered else "unavailable"

        contributors: list[str] = []
        if cnn_used:
            contributors.append("cnn")
        if rules_used:
            contributors.append("rules")

        # One-line attribution explanation for UI.
        if llm_status == "failed":
            expl_line = f"Decision made using {' + '.join(contributors)} due to LLM {llm_failure_reason or 'failure'}"
        elif llm_status == "skipped":
            expl_line = f"Decision made using {' + '.join(contributors)} (LLM skipped)"
        else:
            # LLM used successfully; still note other contributors.
            expl_line = (
                f"Decision made using {' + '.join(contributors + ['llm'])}"
                if contributors
                else "Decision made using LLM"
            )

        decision_metadata = DecisionMetadata(
            decision_source=decision_source,  # type: ignore[arg-type]
            contributors=contributors,
            llm_used=bool(llm_used),
            cnn_used=bool(cnn_used),
            rules_used=bool(rules_used),
            llm_status=llm_status,  # type: ignore[arg-type]
            llm_failure_reason=llm_failure_reason,
            explanation=expl_line,
        )

        # UI contract: "fallback_used" means LLM was not used (rule path) OR LLM failed (fallback path).
        fallback_used = str(decision_source).lower() in ("rule", "fallback")

        # Single memory write: embedding + document + metadata (skip if embedding unusable).
        if embedding_status == "success" and embedding_for_store:
            try:
                embedding = embedding_for_store
                expl_str = _explanation_storage_value(fraud_out.get("explanation"))
                ts = __import__("datetime").datetime.utcnow().isoformat() + "Z"
                preview_b64 = _jpeg_thumbnail_base64(image_bytes) if image_bytes else ""
                self._vector_store.store_claim(
                    claim_id=claim_id,
                    claim_description=claim_description,
                    embedding=embedding,
                    metadata={
                        "claim_id": claim_id,
                        "fraud_score": float(fraud_out.get("fraud_score") or 0.0),
                        "dl_fraud_score": float(dl_fraud_probability) if dl_fraud_probability is not None else 0.0,
                        "decision": str(decision_out.get("decision") or ""),
                        "confidence": float(decision_out.get("confidence_score") or 0.0),
                        "entities": fraud_out.get("entities") or {},
                        "timestamp": ts,
                        "explanation": expl_str,
                        "fraud_signal": fraud_signal or "",
                        "review_status": "",
                        "hitl_needed": hitl.needs_hitl,
                        "hitl_reason": hitl.reason or "",
                        "case_status": "NEW",
                        "assigned_to": "",
                        "assigned_at": "",
                        "updated_at": ts,
                        "image_damage_type": image_damage_type_store,
                        "image_severity": image_severity_store,
                        "cnn_used": "1" if cnn_used else "0",
                        "cnn_label": cnn_label,
                        "cnn_confidence": float(cnn_confidence),
                        "cnn_severity": cnn_severity,
                        "has_image": "1" if image_bytes else "0",
                        "image_preview_base64": preview_b64,
                        "llm_used": "1" if llm_used else "0",
                        "fallback_used": "1" if fallback_used else "0",
                        "decision_source": decision_source,
                        "contributors": ",".join(contributors),
                        "pipeline_json": decision_metadata.model_dump_json(exclude_none=True),
                    },
                )
                logger.info("claim_stored", extra={"claim_id": claim_id})
            except Exception:
                logger.exception("orchestrator_memory_store_failed", extra={"claim_id": claim_id})
        elif embedding_status != "success":
            logger.warning(
                "claim_store_skipped",
                extra={"claim_id": claim_id, "embedding_status": embedding_status},
            )

        metrics.record_claim_processed(hitl_triggered=hitl.needs_hitl)

        total_ms = (time.perf_counter() - workflow_start) * 1000
        logger.info(
            "claim_pipeline_perf",
            extra={
                "claim_id": claim_id,
                "embedding_time_ms": round(embedding_ms, 2),
                "retrieval_time_ms": round(retrieval_ms, 2),
                "image_processing_time_ms": round(image_processing_ms, 2),
                "llm_time_ms": round(llm_ms, 2),
                "dl_time_ms": round(dl_ms, 2),
                "total_time_ms": round(total_ms, 2),
                "parallel_execution": settings.enable_parallel_execution,
                "timeout_triggered": timeout_triggered,
            },
        )
        logger.info(
            "claim_triage_structured",
            extra={
                "claim_id": claim_id,
                "decision": decision_out.get("decision"),
                "confidence": base_confidence,
                "calibrated_confidence": calibrated,
                "hitl_needed": hitl.needs_hitl,
                "hitl_decision_reason": hitl_decision_reason,
                "embedding_status": embedding_status,
                "retrieval_count": retrieval_count,
                "rag_enabled": self._rag_enabled,
                "rag_context_used": rag_context_used,
                "dl_fraud_probability": dl_fraud_probability,
                "fused_fraud_score": decision_out.get("fused_fraud_score"),
                "image_processing_time_ms": round(image_processing_ms, 2),
                "image_features": image_features,
                "embedding_time_ms": round(embedding_ms, 2),
                "retrieval_time_ms": round(retrieval_ms, 2),
                "llm_time_ms": round(llm_ms, 2),
                "dl_time_ms": round(dl_ms, 2),
                "total_time_ms": round(total_ms, 2),
            },
        )
        logger.info(
            "orchestrator_claim_complete",
            extra={
                "duration_ms": round(total_ms, 2),
                "total_time_ms": round(total_ms, 2),
                "decision": decision_out.get("decision"),
            },
        )

        return {
            "claim_id": claim_id,
            "decision": decision_out["decision"],
            "confidence_score": decision_out["confidence_score"],
            "calibrated_confidence": calibrated,
            "hitl_needed": hitl.needs_hitl,
            "review_reason": review_reason,
            "hitl_decision_reason": hitl_decision_reason,
            "llm_used": llm_used,
            "fallback_used": fallback_used,
            "decision_source": decision_source,
            "metadata": decision_metadata.model_dump(exclude_none=True),
            "fraud_signal": fraud_signal,
            "cnn_used": bool(cnn_used),
            "cnn_label": cnn_label,
            "cnn_confidence": float(cnn_confidence),
            "cnn_severity": cnn_severity,
            "agent_outputs": {
                "fraud": fraud_for_client,
                "policy": policy_out,
                "decision": decision_out,
                "dl_fraud": {
                    "enabled": settings.dl_fraud_enabled,
                    "backend": self._dl_fraud_model.backend,
                    "fraud_probability": dl_fraud_probability,
                },
                "image": {
                    "enabled": settings.enable_image_analysis,
                    "model_type": settings.image_model_type,
                    "processing_time_ms": round(image_processing_ms, 2),
                    "features": image_features,
                    "severity_score_used": image_severity_score_for_fusion,
                    "fusion_weight": settings.image_fusion_weight,
                },
            },
        }


def get_insurflow_orchestrator(
    llm_service: LLMService = Depends(get_llm_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    hitl_service: HitlService = Depends(get_hitl_service),
) -> InsurFlowOrchestrator:
    return InsurFlowOrchestrator(
        llm_service=llm_service,
        embedding_service=embedding_service,
        vector_store=vector_store,
        hitl_service=hitl_service,
    )
