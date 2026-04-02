from __future__ import annotations

import json
import logging
from typing import Any, Optional, Tuple

from app.agents.base_agent import BaseAgent
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class FraudAgent(BaseAgent):
    """Fraud signals for micro-insurance claims (LLM-assisted, JSON score + reason)."""

    def __init__(self, *, llm_service: LLMService) -> None:
        super().__init__()
        self._llm_service = llm_service

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "You are a fraud analyst for a micro-insurance product (small sums, high volume, "
            "often mobile-first or instant payouts).\n"
            "Focus on patterns relevant to low-ticket claims: inconsistency, duplication, "
            "implausible amounts vs. cover, missing basics, and abuse of simple products.\n"
            "Analyze the given claim payload and output ONLY valid JSON.\n"
            "The JSON schema is exactly:\n"
            "{\n"
            '  "fraud_score": a number between 0 and 1,\n'
            '  "fraud_reason": a concise explanation (1-3 sentences) grounded in the claim fields.\n'
            "}\n"
            "Rules:\n"
            "- Output JSON only (no markdown, no surrounding text).\n"
            "- fraud_score=0.0 means no fraud signals; 1.0 means strong fraud signals.\n"
            "- Routine legitimate micro-claims should score low unless something is off.\n"
            "- If information is missing, lower fraud_score and explain uncertainty.\n"
        )

        claim_json = json.dumps(input_data, ensure_ascii=False)

        try:
            completion = await self._llm_service.generate(
                prompt=prompt,
                context=f"Claim payload (JSON):\n{claim_json}",
                generation_kwargs={"temperature": 0.2},
            )
        except BaseException as exc:  # noqa: BLE001 - safe structured fallback
            logger.exception("LLM call failed in FraudAgent")
            return {
                "fraud_score": 0.0,
                "fraud_reason": f"LLM call failed: {type(exc).__name__}: {exc}",
            }

        fraud_score, fraud_reason = self._parse_fraud_json(completion.text)
        return {"fraud_score": fraud_score, "fraud_reason": fraud_reason}

    def _parse_fraud_json(self, text: str) -> Tuple[float, str]:
        candidate = self._extract_json_object(text)
        if not candidate:
            return 0.0, "Could not parse LLM response as JSON."

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return 0.0, "LLM response contained invalid JSON."

        fraud_score_raw: Optional[Any] = payload.get("fraud_score")
        reason_raw: Optional[Any] = payload.get("fraud_reason")
        if reason_raw is None:
            reason_raw = payload.get("reason")

        fraud_score = 0.0
        try:
            if isinstance(fraud_score_raw, (int, float, str)):
                fraud_score = float(fraud_score_raw)
        except (TypeError, ValueError):
            fraud_score = 0.0

        fraud_score = min(1.0, max(0.0, fraud_score))

        fraud_reason = "No reason provided by model."
        if isinstance(reason_raw, str) and reason_raw.strip():
            fraud_reason = reason_raw.strip()
        else:
            fraud_reason = "Model did not provide a valid fraud reason field."

        return fraud_score, fraud_reason

    def _extract_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]
