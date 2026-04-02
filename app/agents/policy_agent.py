from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent


class PolicyAgent(BaseAgent):
    """Checks micro-policy coverage: claimed amount must not exceed the policy cap."""

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        raw_amount = input_data.get("claim_amount")
        raw_limit = input_data.get("policy_limit")

        try:
            claim_amount = float(raw_amount) if raw_amount is not None else None
        except (TypeError, ValueError):
            claim_amount = None
        try:
            policy_limit = float(raw_limit) if raw_limit is not None else None
        except (TypeError, ValueError):
            policy_limit = None

        if policy_limit is None or policy_limit <= 0:
            return {
                "policy_valid": False,
                "policy_reason": "Missing or invalid policy_limit; cannot validate coverage.",
            }
        if claim_amount is None or claim_amount < 0:
            return {
                "policy_valid": False,
                "policy_reason": "Missing or invalid claim_amount.",
            }
        if claim_amount > policy_limit:
            return {
                "policy_valid": False,
                "policy_reason": (
                    f"Claim amount {claim_amount} exceeds policy limit {policy_limit}."
                ),
            }
        return {
            "policy_valid": True,
            "policy_reason": (
                f"Claim amount {claim_amount} is within policy limit {policy_limit}."
            ),
        }
