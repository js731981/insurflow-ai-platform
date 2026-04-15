from __future__ import annotations

import logging
from typing import Any, Literal

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

DecisionLiteral = Literal["APPROVED", "REJECTED", "INVESTIGATE"]


def image_severity_to_score(severity: Any) -> float:
    """Map visual severity to a 0–1 fraud-signal style scalar for fusion."""
    s = str(severity or "").strip().lower()
    if s == "high":
        return 0.85
    if s == "medium":
        return 0.5
    if s == "low":
        return 0.22
    return 0.35


# Micro-insurance: above this fraud score, queue for human / secondary review.
FRAUD_INVESTIGATE_THRESHOLD = 0.6
# When similar claims' human majority is APPROVED, require a higher bar to escalate (lean approve).
FRAUD_APPROVED_PATTERN_ESCALATE_THRESHOLD = 0.72
# Regardless of approved pattern, always investigate if fraud signals are this strong.
FRAUD_STRONG_CONTRADICTION_THRESHOLD = 0.84


class DecisionAgent(BaseAgent):
    """Single triage outcome for a micro-insurance claim from fraud + policy signals."""

    def __init__(self, *, fraud_investigate_threshold: float = FRAUD_INVESTIGATE_THRESHOLD) -> None:
        super().__init__()
        self._fraud_threshold = fraud_investigate_threshold

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        fraud_block: dict[str, Any] = input_data.get("fraud") or {}
        policy_block: dict[str, Any] = input_data.get("policy") or {}
        similar_majority = input_data.get("similar_majority_review")
        if similar_majority is not None:
            similar_majority = str(similar_majority).strip().upper() or None
        if similar_majority not in (None, "APPROVED", "REJECTED"):
            similar_majority = None

        fraud_score_raw = fraud_block.get("fraud_score", 0.0)
        try:
            fraud_score = float(fraud_score_raw)
        except (TypeError, ValueError):
            fraud_score = 0.0
        fraud_score = min(1.0, max(0.0, fraud_score))

        image_severity_score = input_data.get("image_severity_score")
        try:
            image_severity_score_f = (
                float(image_severity_score) if image_severity_score is not None else None
            )
        except (TypeError, ValueError):
            image_severity_score_f = None
        if image_severity_score_f is not None:
            image_severity_score_f = min(1.0, max(0.0, image_severity_score_f))

        dl_prob, w_llm_n, w_dl_n, w_img_n, fused = _fuse_fraud_signals(
            fraud_score=fraud_score,
            fraud_probability_dl=input_data.get("fraud_probability_dl"),
            image_severity_score=image_severity_score_f,
            w_llm_raw=input_data.get("dl_fusion_llm_weight"),
            w_dl_raw=input_data.get("dl_fusion_dl_weight"),
            w_image_raw=input_data.get("image_fusion_weight"),
        )

        policy_valid = bool(policy_block.get("policy_valid"))
        policy_reason = str(policy_block.get("policy_reason") or "")
        fraud_explanation = _fraud_explanation_text(fraud_block)

        fusion_note = _fusion_explanation_line(
            fraud_score, dl_prob, image_severity_score_f, fused, w_llm_n, w_dl_n, w_img_n
        )

        if not policy_valid:
            return _finalize_decision_output(
                {
                    "decision": "REJECTED",
                    "confidence_score": 0.9,
                    "explanation": f"Policy check failed. {policy_reason}".strip(),
                },
                fraud_score,
                dl_prob,
                fused,
                w_llm_n,
                w_dl_n,
                w_img_n,
                image_severity_score_f,
            )

        if fused >= FRAUD_STRONG_CONTRADICTION_THRESHOLD:
            conf = min(1.0, max(0.5, fused))
            return _finalize_decision_output(
                {
                    "decision": "INVESTIGATE",
                    "confidence_score": round(conf, 3),
                    "explanation": (
                        f"Strong fraud signals (fused score={fused:.2f}; LLM={fraud_score:.2f}).{fusion_note} "
                        f"{fraud_explanation}".strip()
                    ),
                },
                fraud_score,
                dl_prob,
                fused,
                w_llm_n,
                w_dl_n,
                w_img_n,
                image_severity_score_f,
            )

        escalate_threshold = self._fraud_threshold
        pattern_note = ""
        if similar_majority == "APPROVED":
            escalate_threshold = FRAUD_APPROVED_PATTERN_ESCALATE_THRESHOLD
            pattern_note = " Similar reviewed claims were mostly approved; higher bar to escalate."

        if fused >= escalate_threshold:
            conf = min(1.0, max(0.5, fused))
            return _finalize_decision_output(
                {
                    "decision": "INVESTIGATE",
                    "confidence_score": round(conf, 3),
                    "explanation": (
                        f"Elevated fraud signals (fused score={fused:.2f}; LLM={fraud_score:.2f}).{pattern_note}"
                        f" {fusion_note} {fraud_explanation}".strip()
                    ),
                },
                fraud_score,
                dl_prob,
                fused,
                w_llm_n,
                w_dl_n,
                w_img_n,
                image_severity_score_f,
            )

        conf = min(1.0, max(0.5, 1.0 - fused))
        approve_note = ""
        if similar_majority == "APPROVED":
            approve_note = (
                " Reviewed similar claims were mostly approved; leaning approve without strong fraud contradiction."
            )
            conf = min(1.0, conf * 1.06)
        expl = (
            f"Policy valid and fused fraud score acceptable ({fused:.2f}; LLM={fraud_score:.2f}).{approve_note}"
            f"{fusion_note} {fraud_explanation}".strip()
        )
        return _finalize_decision_output(
            {
                "decision": "APPROVED",
                "confidence_score": round(conf, 3),
                "explanation": expl.strip(),
            },
            fraud_score,
            dl_prob,
            fused,
            w_llm_n,
            w_dl_n,
            w_img_n,
            image_severity_score_f,
        )


def _fuse_fraud_signals(
    *,
    fraud_score: float,
    fraud_probability_dl: Any,
    image_severity_score: float | None,
    w_llm_raw: Any,
    w_dl_raw: Any,
    w_image_raw: Any,
) -> tuple[float | None, float, float, float, float]:
    """Returns (dl_probability or None, n_llm, n_dl, n_image, fused_score)."""
    try:
        w_llm = float(w_llm_raw) if w_llm_raw is not None else 0.7
    except (TypeError, ValueError):
        w_llm = 0.7
    try:
        w_dl = float(w_dl_raw) if w_dl_raw is not None else 0.3
    except (TypeError, ValueError):
        w_dl = 0.3
    try:
        w_img = float(w_image_raw) if w_image_raw is not None else 0.2
    except (TypeError, ValueError):
        w_img = 0.2

    dl_val: float | None
    try:
        if fraud_probability_dl is None:
            dl_val = None
        else:
            dl_val = float(fraud_probability_dl)
            dl_val = min(1.0, max(0.0, dl_val))
    except (TypeError, ValueError):
        dl_val = None

    img_val: float | None
    try:
        if image_severity_score is None:
            img_val = None
        else:
            img_val = float(image_severity_score)
            img_val = min(1.0, max(0.0, img_val))
    except (TypeError, ValueError):
        img_val = None

    use_llm = max(0.0, w_llm)
    use_dl = max(0.0, w_dl) if dl_val is not None else 0.0
    use_img = max(0.0, w_img) if img_val is not None else 0.0
    total = use_llm + use_dl + use_img
    if total <= 0:
        return dl_val, 1.0, 0.0, 0.0, fraud_score

    n_llm = use_llm / total
    n_dl = use_dl / total
    n_img = use_img / total
    fused = n_llm * fraud_score
    if dl_val is not None:
        fused += n_dl * dl_val
    if img_val is not None:
        fused += n_img * img_val
    return dl_val, n_llm, n_dl, n_img, min(1.0, max(0.0, fused))


def _fusion_explanation_line(
    fraud_score: float,
    dl_prob: float | None,
    img_prob: float | None,
    fused: float,
    w_llm_n: float,
    w_dl_n: float,
    w_img_n: float,
) -> str:
    parts = [f" Hybrid fusion: fused={fused:.2f} (LLM={fraud_score:.2f}×{w_llm_n:.2f}"]
    if dl_prob is not None:
        parts.append(f" + DL={dl_prob:.2f}×{w_dl_n:.2f}")
    if img_prob is not None:
        parts.append(f" + Image={img_prob:.2f}×{w_img_n:.2f}")
    parts.append(").")
    if dl_prob is None and img_prob is None:
        return ""
    return "".join(parts)


def _attach_fusion_fields(
    out: dict[str, Any],
    fraud_score: float,
    dl_prob: float | None,
    fused: float,
    w_llm_n: float,
    w_dl_n: float,
    w_img_n: float,
    image_severity_score: float | None,
) -> None:
    out["fused_fraud_score"] = round(fused, 4)
    out["fraud_score_llm"] = round(fraud_score, 4)
    out["fraud_probability_dl"] = None if dl_prob is None else round(dl_prob, 4)
    out["image_severity_score"] = None if image_severity_score is None else round(image_severity_score, 4)
    out["fraud_fusion_weights"] = {
        "llm": round(w_llm_n, 4),
        "dl": round(w_dl_n, 4),
        "image": round(w_img_n, 4),
    }


def _finalize_decision_output(
    out: dict[str, Any],
    fraud_score: float,
    dl_prob: float | None,
    fused: float,
    w_llm_n: float,
    w_dl_n: float,
    w_img_n: float,
    image_severity_score: float | None,
) -> dict[str, Any]:
    _attach_fusion_fields(out, fraud_score, dl_prob, fused, w_llm_n, w_dl_n, w_img_n, image_severity_score)
    logger.info(
        "decision_fusion_contribution",
        extra={
            "decision": out["decision"],
            "fraud_score_llm": fraud_score,
            "fraud_probability_dl": dl_prob,
            "image_severity_score": image_severity_score,
            "fused_fraud_score": fused,
            "fusion_llm_weight": w_llm_n,
            "fusion_dl_weight": w_dl_n,
            "fusion_image_weight": w_img_n,
        },
    )
    return out


def _fraud_explanation_text(fraud_block: dict[str, Any]) -> str:
    raw = fraud_block.get("explanation")
    if isinstance(raw, dict):
        summary = str(raw.get("summary") or "").strip()
        factors = raw.get("key_factors")
        lines: list[str] = []
        if summary:
            lines.append(summary)
        if isinstance(factors, list):
            for f in factors[:4]:
                t = str(f).strip()
                if t:
                    lines.append(f"- {t}")
        ref = str(raw.get("similar_case_reference") or "").strip()
        if ref:
            lines.append(f"Similar cases: {ref}")
        if lines:
            return "\n".join(lines)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return str(fraud_block.get("fraud_reason") or "").strip()
