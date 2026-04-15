from __future__ import annotations

import logging
from typing import Any

from app.services.vector_store import SimilarHit

logger = logging.getLogger(__name__)


def _human_review_label(meta: dict[str, Any]) -> str:
    rs = str(meta.get("review_status") or "").strip()
    if rs:
        return rs
    return str(meta.get("reviewed_action") or "").strip()


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _topic_blurb(document: str, *, max_chars: int = 72) -> str:
    t = " ".join((document or "").strip().split())
    if not t:
        return "similar claim"
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars].rsplit(" ", 1)[0]
    base = (cut or t[:max_chars]).rstrip()
    return f"{base}…" if len(t) > max_chars else base


def _decision_word(meta: dict[str, Any]) -> str:
    d = str(meta.get("decision") or "").strip().lower()
    if d in ("approved", "rejected", "investigate"):
        return d
    return "unknown"


def _fraud_risk_phrase(meta: dict[str, Any]) -> str:
    try:
        fs = float(meta.get("fraud_score") or 0.0)
    except (TypeError, ValueError):
        fs = 0.0
    if fs < 0.35:
        return "low fraud risk"
    if fs < 0.65:
        return "medium fraud risk"
    return "high fraud risk"


def _review_suffix(meta: dict[str, Any]) -> str:
    if not (
        str(meta.get("review_status") or "").strip()
        or str(meta.get("reviewed_action") or "").strip()
    ):
        return ""
    lab = _human_review_label(meta)
    if not lab:
        return ""
    short = lab.strip()[:40]
    return f", reviewed: {short}"


class ContextBuilder:
    """Turns retrieval hits into a short LLM context (no raw metadata JSON)."""

    def __init__(self, *, max_tokens: int = 256) -> None:
        self._max_tokens = max(64, min(2048, int(max_tokens)))

    def build(self, hits: list[SimilarHit]) -> str:
        if not hits:
            return ""

        # Keep this ultra-compact: 1 line per claim, no extra JSON/metadata blocks.
        lines: list[str] = ["Similar claims:"]
        for idx, h in enumerate(hits, start=1):
            blurb = _topic_blurb(h.document or "")
            dec = _decision_word(h.metadata).replace("approved", "approved").replace("rejected", "rejected")
            # Example target:
            # 1. Minor crack → approved
            tail = _review_suffix(h.metadata)
            arrow = "→"
            lines.append(f"{idx}. {blurb} {arrow} {dec}{tail}")

        text = "\n".join(lines).strip()

        max_chars = self._max_tokens * 4
        while text and _approx_tokens(text) > self._max_tokens:
            if "\n\n" not in text:
                text = text[:max_chars].rstrip()
                break
            text = text.rsplit("\n\n", 1)[0].strip()

        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n[context truncated]"

        logger.debug(
            "context_builder_built",
            extra={"hits": len(hits), "approx_tokens": _approx_tokens(text)},
        )
        return text
