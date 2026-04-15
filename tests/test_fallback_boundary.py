from __future__ import annotations

import asyncio
from io import BytesIO

import pytest
from PIL import Image

from app.agents.orchestrator import InsurFlowOrchestrator
from app.core.config import settings
from app.services.hitl_service import HitlService


def _png_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _DummyEmbeddingService:
    async def embed(self, _text: str):
        return None


class _DummyVectorStore:
    def query_similar_hits(self, *, query_embedding, exclude_claim_id=None, n_results=3, where=None):
        return []

    def store_claim(self, *args, **kwargs):
        return None


class _DummyLLMService:
    async def generate(self, *args, **kwargs):
        raise RuntimeError("LLM should not be called in this test")


@pytest.mark.asyncio
async def test_fallback_boundary_claim_amount_equals_policy_limit_is_approved(monkeypatch):
    # Ensure we exercise the fallback (LLM failure/timeout) path deterministically.
    monkeypatch.setattr(settings, "enable_parallel_execution", False)
    monkeypatch.setattr(settings, "claim_timeout_s", 1.0)
    monkeypatch.setattr(settings, "rag_enabled", False)

    orch = InsurFlowOrchestrator(
        llm_service=_DummyLLMService(),
        embedding_service=_DummyEmbeddingService(),
        vector_store=_DummyVectorStore(),
        hitl_service=HitlService(approve_confidence_threshold=0.0),  # keep HITL out of the assertion
    )

    async def _fake_llm_timeout(_input):
        await asyncio.sleep(0.01)
        return {
            "fraud_score": 0.5,
            "decision": "INVESTIGATE",
            "confidence": 0.5,
            "entities": {},
            "explanation": {
                "summary": "LLM timed out (test)",
                "key_factors": ["LLM timeout", "Fallback path"],
                "similar_case_reference": "",
            },
            "_llm_failed": True,
            "_timeout_triggered": True,
            "_llm_latency_ms": 10,
        }

    monkeypatch.setattr(orch._fraud_agent, "run", _fake_llm_timeout)
    monkeypatch.setattr(
        orch._image_service,
        "analyze",
        lambda _b: {"damage_type": "screen", "severity": "low", "confidence": 1.0, "backend": "test"},
    )

    result = await orch.process_claim(
        {
            "claim_id": "BOUNDARY-001",
            "claim_amount": 1000,
            "policy_limit": 1000,
            "description": "Boundary test",
            "_image_bytes": _png_bytes(),
        }
    )

    assert result["decision"] == "APPROVED"
    assert result["hitl_needed"] is False


@pytest.mark.asyncio
async def test_low_severity_fallback_within_policy_limit_does_not_require_hitl(monkeypatch):
    # Ensure we exercise the fallback (LLM failure/timeout) path deterministically.
    monkeypatch.setattr(settings, "enable_parallel_execution", False)
    monkeypatch.setattr(settings, "claim_timeout_s", 1.0)
    monkeypatch.setattr(settings, "rag_enabled", False)

    orch = InsurFlowOrchestrator(
        llm_service=_DummyLLMService(),
        embedding_service=_DummyEmbeddingService(),
        vector_store=_DummyVectorStore(),
        hitl_service=HitlService(approve_confidence_threshold=0.7),
    )

    async def _fake_llm_timeout(_input):
        await asyncio.sleep(0.01)
        return {
            "fraud_score": 0.5,
            "decision": "INVESTIGATE",
            "confidence": 0.5,
            "entities": {},
            "explanation": {
                "summary": "LLM timed out (test)",
                "key_factors": ["LLM timeout", "Fallback path"],
                "similar_case_reference": "",
            },
            "_llm_failed": True,
            "_timeout_triggered": True,
            "_llm_latency_ms": 10,
        }

    monkeypatch.setattr(orch._fraud_agent, "run", _fake_llm_timeout)
    monkeypatch.setattr(
        orch._image_service,
        "analyze",
        lambda _b: {"damage_type": "screen", "severity": "low", "confidence": 1.0, "backend": "test"},
    )

    result = await orch.process_claim(
        {
            "claim_id": "LOW-OK-001",
            "claim_amount": 800,
            "policy_limit": 1000,
            "description": "Low severity, within limit",
            "_image_bytes": _png_bytes(),
        }
    )

    assert result["decision"] == "APPROVED"
    assert result["hitl_needed"] is False


@pytest.mark.asyncio
async def test_description_image_mismatch_overrides_approval_to_investigate(monkeypatch):
    # Exercise the fallback approval path but ensure mismatch rule overrides it.
    monkeypatch.setattr(settings, "enable_parallel_execution", False)
    monkeypatch.setattr(settings, "claim_timeout_s", 1.0)
    monkeypatch.setattr(settings, "rag_enabled", False)
    monkeypatch.setattr(settings, "enable_image_analysis", True)

    orch = InsurFlowOrchestrator(
        llm_service=_DummyLLMService(),
        embedding_service=_DummyEmbeddingService(),
        vector_store=_DummyVectorStore(),
        hitl_service=HitlService(approve_confidence_threshold=0.0),  # keep HITL out of threshold effects
    )

    async def _fake_llm_timeout(_input):
        await asyncio.sleep(0.01)
        return {
            "fraud_score": 0.5,
            "decision": "INVESTIGATE",
            "confidence": 0.5,
            "entities": {},
            "explanation": {
                "summary": "LLM timed out (test)",
                "key_factors": ["LLM timeout", "Fallback path"],
                "similar_case_reference": "",
            },
            "_llm_failed": True,
            "_timeout_triggered": True,
            "_llm_latency_ms": 10,
        }

    monkeypatch.setattr(orch._fraud_agent, "run", _fake_llm_timeout)
    # Force CNN label to "no_damage" to create the mismatch with a "crack" description.
    monkeypatch.setattr(
        orch._image_cnn_service,
        "analyze",
        lambda _b: {
            "label": "no_damage",
            "confidence": 0.95,
            "severity": "low",
            "damage_type": "no_damage",
            "backend": "test",
            "signals": {"cnn_label": "no_damage", "cnn_confidence": 0.95, "cnn_used": True},
        },
    )

    result = await orch.process_claim(
        {
            "claim_id": "MISMATCH-001",
            "claim_amount": 100,
            "policy_limit": 1000,
            "description": "Screen cracked after drop",
            "_image_bytes": _png_bytes(),
        }
    )

    assert result["decision"] == "INVESTIGATE"
    assert result.get("fraud_signal") == "image_text_mismatch"
