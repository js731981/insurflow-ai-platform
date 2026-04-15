"""HTML / text formatting for the HF demo UI."""
from __future__ import annotations

import html
import json
import re
from typing import Any, Optional


def risk_zone(score: float) -> tuple[str, str]:
    """UI fraud visualization bands: 0–0.3 LOW, 0.3–0.7 MEDIUM, 0.7–1 HIGH."""
    s = max(0.0, min(1.0, float(score)))
    if s < 0.3:
        return ("LOW RISK", "ok")
    if s < 0.7:
        return ("MEDIUM RISK", "warn")
    return ("HIGH RISK", "bad")


def decision_pill(decision: str) -> tuple[str, str]:
    d = (decision or "").upper().strip()
    mapping = {
        "APPROVE": ("APPROVED", "ok"),
        "APPROVED": ("APPROVED", "ok"),
        "INVESTIGATE": ("INVESTIGATE", "warn"),
        "REJECT": ("REJECTED", "bad"),
        "REJECTED": ("REJECTED", "bad"),
    }
    return mapping.get(d, (d or "—", "neutral"))


def prettify_label(label: str) -> str:
    return (label or "").replace("_", " ").strip().title()


severity_map: dict[str, str] = {
    "no_damage": "LOW",
    "minor_crack": "MEDIUM",
    "major_crack": "HIGH",
}


def severity_from_cnn_label(cnn_label: str) -> str:
    """
    Map model label -> severity text shown in UI.
    Never returns UNKNOWN; defaults to MEDIUM for safety/readability.
    """
    key = (cnn_label or "").strip().lower()
    return severity_map.get(key) or "MEDIUM"


def format_explanation(data: dict) -> list[str]:
    """
    Human-readable, user-facing explanation bullets.
    Designed to replace raw/system-style key-value output.
    """
    claim = data.get("claim") if isinstance(data.get("claim"), dict) else {}
    amount = claim.get("claim_amount", data.get("claim_amount"))
    limit = claim.get("policy_limit", data.get("policy_limit"))
    try:
        amount_f = float(amount or 0.0)
    except (TypeError, ValueError):
        amount_f = 0.0
    try:
        limit_f = float(limit or 0.0)
    except (TypeError, ValueError):
        limit_f = 0.0

    cnn_label = str(data.get("cnn_label") or data.get("label") or "").strip().lower()
    pipeline = data.get("pipeline") if isinstance(data.get("pipeline"), list) else []
    llm_status = ""
    for st in pipeline:
        if isinstance(st, dict) and str(st.get("id") or "").lower() == "llm":
            llm_status = str(st.get("status") or "").lower().strip()
            break

    # Fraud score -> qualitative phrase
    try:
        fs = float(data.get("fraud_score") or 0.0)
    except (TypeError, ValueError):
        fs = 0.0
    band, _ = risk_zone(fs)
    fraud_phrase = (
        "low likelihood of fraud"
        if band.startswith("LOW")
        else ("moderate likelihood of fraud" if band.startswith("MEDIUM") else "high likelihood of fraud")
    )

    bullets: list[str] = []

    if limit_f > 0:
        within = amount_f <= limit_f
        bullets.append(
            f"Claim amount (${amount_f:,.0f}) is {'within' if within else 'above'} the policy limit (${limit_f:,.0f})."
        )
    else:
        bullets.append(f"Claim amount is ${amount_f:,.0f} (no policy limit provided).")

    if cnn_label and cnn_label not in {"unknown", "n/a", "none"}:
        sev = severity_from_cnn_label(cnn_label)
        bullets.append(
            f"Image analysis detected {prettify_label(cnn_label).lower()} ({sev.lower()} severity)."
        )
    else:
        bullets.append("No image was provided, so the decision relied on non-visual signals.")

    bullets.append(f"Risk signals indicate {fraud_phrase}.")

    if llm_status in {"skipped", "failed"}:
        bullets.append("Decision made using rule-based evaluation due to latency optimisation.")

    return bullets

def pill_html(label: str, cls: str) -> str:
    extra = " badge-green" if cls == "ok" else ""
    return f"<span class='pill {cls}{extra}'>{html.escape(label)}</span>"


def fraud_score_panel_html(score: float) -> str:
    band, band_cls = risk_zone(score)
    pct = max(0.0, min(1.0, float(score))) * 100.0
    color = "#22c55e" if band_cls == "ok" else ("#f59e0b" if band_cls == "warn" else "#ef4444")
    dash = 251.2 * (1.0 - float(score))  # 2*pi*r for r=40
    return f"""
    <div class="kpi fade-in fraud-kpi">
      <div style="flex:1;min-width:0;">
        <div class="label">Fraud score (0–1)</div>
        <div class="fraud-score-row">
          <div class="gauge-wrap" aria-hidden="true">
            <svg class="gauge-svg" viewBox="0 0 100 100">
              <circle class="gauge-bg" cx="50" cy="50" r="40" />
              <circle class="gauge-fg" cx="50" cy="50" r="40"
                stroke="{html.escape(color)}"
                stroke-dasharray="251.2"
                stroke-dashoffset="{dash:.2f}" />
            </svg>
            <div class="gauge-center">{html.escape(f"{float(score):.2f}")}</div>
          </div>
          <div style="flex:1;min-width:0;">
            <div class="risk-bar-label">Relative exposure</div>
            <div class="risk-bar-track"><div class="risk-bar-fill zone-{band_cls}" style="width:{pct:.1f}%"></div></div>
            <div class="help-text">Higher values suggest more review before payout.</div>
          </div>
        </div>
      </div>
      <div class="kpi-side">
        {pill_html(band, band_cls)}
      </div>
    </div>
    """


def breakdown_panel_html(breakdown: dict[str, Any]) -> str:
    rows = []
    for key, lab in (("cnn", "CNN"), ("rules", "Rules"), ("llm", "LLM")):
        try:
            v = float(breakdown.get(key, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        mag = min(1.0, max(0.0, abs(v)))
        rows.append(
            f"""
            <div class="breakdown-row">
              <div class="breakdown-name">{html.escape(lab)}</div>
              <div class="breakdown-val">{html.escape(f'{v:+.2f}')}</div>
              <div class="breakdown-bar-track">
                <div class="breakdown-bar-fill {'neg' if v < 0 else 'pos'}" style="width:{mag*100:.0f}%"></div>
              </div>
            </div>
            """
        )
    inner = "\n".join(rows)
    return f"""
    <div class="soft-card fade-in">
      <div class="section-h">Decision breakdown</div>
      <div class="help-text" style="margin-bottom:10px;">Contribution-style signals (demo attribution).</div>
      {inner}
    </div>
    """


def pipeline_panel_html(steps: list[dict[str, str]]) -> str:
    parts = []
    sub = {
        "cnn": "Image classification",
        "rules": "Business validation",
        "llm": "Fraud reasoning",
        "decision": "Final outcome",
        "image": "Claim evidence",
    }
    for i, st in enumerate(steps):
        status = str(st.get("status") or "skipped").lower()
        cls = {"used": "st-used", "skipped": "st-skip", "failed": "st-fail"}.get(status, "st-skip")
        raw_id = str(st.get("id") or "").lower()
        label = html.escape(str(st.get("label") or st.get("id") or "—"))
        sublabel = html.escape(sub.get(raw_id, ""))
        badge = status.upper()
        parts.append(
            f"""
            <div class="pipe-step {cls}">
              <div class="pipe-step-label">{label}</div>
              <div class="pipe-step-sublabel">{sublabel}</div>
              <div class="pipe-step-status">{html.escape(badge)}</div>
            </div>
            """
        )
        if i < len(steps) - 1:
            parts.append('<div class="pipe-arrow" aria-hidden="true">→</div>')
    return f"""
    <div class="soft-card fade-in">
      <div class="section-h">AI pipeline</div>
      <div class="pipe-flow">{"".join(parts)}</div>
      <div class="pipe-legend">
        <span><span class="dot st-used"></span> USED</span>
        <span><span class="dot st-skip"></span> SKIPPED</span>
        <span><span class="dot st-fail"></span> FAILED</span>
      </div>
    </div>
    """


def inconsistency_banner_html(message: str) -> str:
    if not message.strip():
        return ""
    return f"""
    <div class="banner-warn fade-in" role="alert">
      <div class="banner-title">⚠️ INCONSISTENT CLAIM (IMAGE VS DESCRIPTION)</div>
      <div class="banner-body">{html.escape(message)}</div>
    </div>
    """


def latency_panel_html(times: dict[str, Any], pipeline: Optional[list[dict[str, Any]]] = None) -> str:
    def g(k: str) -> str:
        try:
            return f"{float(times.get(k, 0.0)):.1f}"
        except (TypeError, ValueError):
            return "—"

    # Realism fix: avoid showing 0.0ms or fake numbers when LLM was skipped.
    llm_status = ""
    if isinstance(pipeline, list):
        for st in pipeline:
            if isinstance(st, dict) and str(st.get("id") or "").lower() == "llm":
                llm_status = str(st.get("status") or "").lower().strip()
                break

    llm_display: str
    if llm_status in {"skipped", "failed"}:
        llm_display = "LLM skipped (latency optimisation)"
    else:
        try:
            llm_v = float(times.get("llm", 0.0))
            llm_display = "LLM skipped (latency optimisation)" if llm_v == 0.0 else f"{llm_v:.1f} ms"
        except (TypeError, ValueError):
            llm_display = "—"

    return f"""
    <div class="soft-card fade-in latency-card">
      <div class="section-h">Latency</div>
      <div class="latency-grid">
        <div><span class="lat-label">Total</span><span class="lat-val">{g('total')} ms</span></div>
        <div><span class="lat-label">CNN / vision</span><span class="lat-val">{g('cnn')} ms</span></div>
        <div><span class="lat-label">LLM</span><span class="lat-val">{html.escape(llm_display)}</span></div>
      </div>
    </div>
    """


def analytics_panel_html(summary: Optional[dict[str, Any]], error: Optional[str] = None) -> str:
    if error:
        return f"<div class='soft-card'><div class='section-h'>Analytics</div><div class='help-text'>{html.escape(error)}</div></div>"
    if not summary:
        return "<div class='soft-card'><div class='section-h'>Analytics</div><div class='help-text'>No data.</div></div>"
    total = int(summary.get("total_claims") or 0)
    dist = summary.get("decision_distribution") if isinstance(summary.get("decision_distribution"), dict) else {}
    appr = int(dist.get("APPROVED") or 0)
    inv = int(dist.get("INVESTIGATE") or 0)
    ar = round((appr / total) * 100.0, 1) if total else 0.0
    ir = round((inv / total) * 100.0, 1) if total else 0.0
    return f"""
    <div class="soft-card fade-in analytics-card">
      <div class="section-h">Mini analytics</div>
      <div class="analytics-grid">
        <div class="analytics-tile"><div class="analytics-num">{total}</div><div class="analytics-cap">Total claims</div></div>
        <div class="analytics-tile"><div class="analytics-num">{ar}%</div><div class="analytics-cap">Approval rate</div></div>
        <div class="analytics-tile"><div class="analytics-num">{ir}%</div><div class="analytics-cap">Investigation rate</div></div>
      </div>
    </div>
    """


def explanation_card_html(explanation: Any) -> str:
    def _inline_code(s: str) -> str:
        # Convert simple backtick spans to <code> while keeping everything escaped.
        parts = (s or "").split("`")
        out: list[str] = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                out.append(html.escape(part))
            else:
                out.append(f"<code>{html.escape(part)}</code>")
        return "".join(out)

    items: list[str] = []
    if isinstance(explanation, list):
        items = [str(x).strip() for x in explanation if str(x).strip()][:10]
    else:
        raw_lines = [ln.strip() for ln in str(explanation or "").splitlines()]
        for ln in raw_lines:
            if not ln:
                continue
            if ln.startswith("- "):
                items.append(ln[2:].strip())
            elif ln.startswith("  - "):
                items.append(ln[4:].strip())
        if not items:
            items = [ln for ln in raw_lines if ln][:10]
        items = items[:10]
    bullets = "\n".join([f"<li>{_inline_code(it)}</li>" for it in items])
    return f"""
    <div class="soft-card fade-in">
      <div class="section-h">Explanation</div>
      <div class="explanation-box">
        <ul class="expl-list">{bullets}</ul>
      </div>
      <div class="help-text" style="margin-top:10px;font-size:12px;">
        Demo disclaimer: outputs are illustrative and not an insurance decision.
      </div>
    </div>
    """


def model_insights_html(cnn_label: str, severity: str, backend_note: str = "") -> str:
    pretty = prettify_label(cnn_label) if cnn_label else "—"
    sev_src = (severity or "").strip()
    sev_ui = sev_src.upper() if sev_src else severity_from_cnn_label(cnn_label)
    sev_ui = sev_ui or "MEDIUM"
    sev_key = sev_ui.lower()
    cls = {"low": "ok", "medium": "warn", "high": "bad", "n/a": "neutral"}.get(sev_key, "warn")
    sev_label = f"Severity: {prettify_label(sev_ui)}"
    note = f"<div class='help-text' style='margin-top:10px;font-size:13px;'>{html.escape(backend_note)}</div>" if backend_note else ""
    return f"""
    <div class="soft-card fade-in">
      <div class="section-h">Model insights</div>
      <div class="pill-row">
        {pill_html(f"Label: {pretty}", "neutral")}
        {pill_html(sev_label, cls)}
      </div>
      {note}
    </div>
    """


def decision_card_html(decision: str, context_tag: str | None = None) -> str:
    label, cls = decision_pill(decision)
    tag = f"<div class='decision-tag'>{html.escape(context_tag)}</div>" if context_tag else ""
    return f"""
    <div class="soft-card fade-in">
      <div class="section-h">Decision</div>
      <div class="pill-row">{pill_html(label, cls)}</div>
      {tag}
    </div>
    """


def build_report_dict(
    *,
    claim_id: str,
    description: str,
    claim_amount: float,
    policy_limit: float,
    decision: str,
    explanation: str,
    cnn_label: str,
    fraud_score: float,
    source: str,
    breakdown: Optional[dict[str, Any]] = None,
    pipeline: Optional[list[dict[str, Any]]] = None,
    latency_ms: Optional[dict[str, Any]] = None,
    fraud_signal: Optional[str] = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "claim": {
            "claim_id": claim_id,
            "description": description,
            "claim_amount": claim_amount,
            "policy_limit": policy_limit,
        },
        "decision": decision,
        "fraud_score": fraud_score,
        "cnn_label": cnn_label,
        "explanation": explanation,
        "source": source,
    }
    if breakdown is not None:
        out["breakdown"] = breakdown
    if pipeline is not None:
        out["pipeline"] = pipeline
    if latency_ms is not None:
        out["latency_ms"] = latency_ms
    if fraud_signal:
        out["fraud_signal"] = fraud_signal
    return out


def report_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
