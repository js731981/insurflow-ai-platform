from __future__ import annotations

from typing import Any, Literal

from app.agents.base_agent import BaseAgent

DecisionLiteral = Literal["APPROVED", "REJECTED", "INVESTIGATE"]

# Micro-insurance: above this fraud score, queue for human / secondary review.
FRAUD_INVESTIGATE_THRESHOLD = 0.6


class DecisionAgent(BaseAgent):
    """Single triage outcome for a micro-insurance claim from fraud + policy signals."""

    def __init__(self, *, fraud_investigate_threshold: float = FRAUD_INVESTIGATE_THRESHOLD) -> None:
        super().__init__()
        self._fraud_threshold = fraud_investigate_threshold

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        fraud_block: dict[str, Any] = input_data.get("fraud") or {}
        policy_block: dict[str, Any] = input_data.get("policy") or {}

        fraud_score_raw = fraud_block.get("fraud_score", 0.0)
        try:
            fraud_score = float(fraud_score_raw)
        except (TypeError, ValueError):
            fraud_score = 0.0
        fraud_score = min(1.0, max(0.0, fraud_score))

        policy_valid = bool(policy_block.get("policy_valid"))
        policy_reason = str(policy_block.get("policy_reason") or "")
        fraud_reason = str(fraud_block.get("fraud_reason") or "")

        if not policy_valid:
            return {
                "decision": "REJECTED",
                "confidence_score": 0.9,
                "explanation": f"Policy check failed. {policy_reason}".strip(),
            }

        if fraud_score >= self._fraud_threshold:
            conf = min(1.0, max(0.5, fraud_score))
            return {
                "decision": "INVESTIGATE",
                "confidence_score": round(conf, 3),
                "explanation": (
                    f"Elevated fraud signals (score={fraud_score:.2f}). {fraud_reason}".strip()
                ),
            }

        conf = min(1.0, max(0.5, 1.0 - fraud_score))
        return {
            "decision": "APPROVED",
            "confidence_score": round(conf, 3),
            "explanation": (
                f"Policy valid and fraud score acceptable ({fraud_score:.2f}). "
                f"{fraud_reason}".strip()
            ),
        }
