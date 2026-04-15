from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HitlDecision:
    needs_hitl: bool
    reason: str | None = None


class HitlService:
    """Human-in-the-loop policy (kept out of agents)."""

    def __init__(self, *, approve_confidence_threshold: float = 0.7) -> None:
        # Business rule: auto-approve only if confidence clears this bar.
        self._approve_confidence_threshold = float(approve_confidence_threshold)

    def evaluate(self, *, decision: str, confidence: float) -> HitlDecision:
        d = (decision or "").strip().upper()
        try:
            c = float(confidence)
        except (TypeError, ValueError):
            c = 0.0
        c = min(1.0, max(0.0, c))

        if d == "INVESTIGATE":
            return HitlDecision(needs_hitl=True, reason="Decision escalated to INVESTIGATE.")
        if d == "REJECTED":
            return HitlDecision(needs_hitl=False, reason="Not required (rejected).")

        # Default to safety if decision is unknown.
        if d != "APPROVED":
            return HitlDecision(needs_hitl=True, reason="Unknown decision; routed for human review.")

        if c >= self._approve_confidence_threshold:
            return HitlDecision(needs_hitl=False, reason="Not required (approved with sufficient confidence).")
        return HitlDecision(
            needs_hitl=True,
            reason=f"Low confidence (< {self._approve_confidence_threshold}).",
        )

