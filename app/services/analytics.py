from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from app.services.vector_store import VectorStore

DECISION_KEYS = frozenset({"APPROVED", "REJECTED", "INVESTIGATE"})
REVIEW_KEYS = frozenset({"APPROVED", "REJECTED"})


def _fetch_all_claim_rows(vector_store: VectorStore) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    batch = 500
    while True:
        chunk = vector_store.list_claims(limit=batch, offset=offset)
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return out


def _parse_entities_json(meta: dict[str, Any]) -> dict[str, Any]:
    ej = meta.get("entities_json")
    if not isinstance(ej, str) or not ej.strip():
        return {}
    try:
        o = json.loads(ej)
        return o if isinstance(o, dict) else {}
    except json.JSONDecodeError:
        return {}


def _safe_fraud_score(meta: dict[str, Any]) -> Optional[float]:
    raw = meta.get("fraud_score")
    if raw is None or raw == "":
        return None
    try:
        x = float(raw)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _normalize_decision(meta: dict[str, Any]) -> Optional[str]:
    d = str(meta.get("decision") or "").strip().upper()
    return d if d in DECISION_KEYS else None


def _normalize_review_bucket(meta: dict[str, Any]) -> str:
    rs = str(meta.get("review_status") or "").strip().upper()
    if rs in REVIEW_KEYS:
        return rs
    ra = str(meta.get("reviewed_action") or "").strip().upper()
    if ra in REVIEW_KEYS:
        return ra
    return "PENDING"


def _timestamp_to_day(meta: dict[str, Any]) -> Optional[str]:
    ts = meta.get("timestamp")
    if ts is None:
        return None
    s = str(ts).strip()
    if not s:
        return None
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    try:
        iso = s.replace("Z", "+00:00") if s.endswith("Z") else s
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt.date().isoformat()
    except (ValueError, TypeError):
        return None


def _scalar_entity_value(entities: dict[str, Any], key: str) -> str:
    v = entities.get(key)
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float, bool)):
        return str(v).strip()
    return ""


def _top_kv(counter: Counter[str], *, limit: int = 8) -> list[dict[str, Any]]:
    items = counter.most_common(limit)
    return [{"value": k, "count": int(c)} for k, c in items if k]


FRAUD_CLUSTER_SCORE_MIN = 0.7
FRAUD_CLUSTER_MIN_CLAIMS = 3
DECISION_RATIO_MIN_TOTAL = 5
INVESTIGATE_RATIO_ALERT = 0.30
REJECTION_RATIO_ALERT = 0.30
REPEAT_PRODUCT_MIN_TOTAL = 6
REPEAT_PRODUCT_MIN_COUNT = 4
REPEAT_PRODUCT_SHARE = 0.28
REPEAT_DESC_MIN_SAME = 3
DESC_WORD_MIN_LEN = 4
DESC_SIGNATURE_MAX_WORDS = 8
HITL_RATIO_ALERT = 0.35
HITL_MIN_TOTAL = 5
TREND_WINDOW = 12
TREND_MIN_PREV = 5
TREND_SURGE_FACTOR = 2.0
TREND_RECENT_RATIO = 0.25


def _hitl_needed(meta: dict[str, Any]) -> bool:
    v = meta.get("hitl_needed")
    if v is True:
        return True
    if v is False or v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes")


def _parse_meta_datetime(meta: dict[str, Any]) -> Optional[datetime]:
    ts = meta.get("timestamp")
    if ts is None:
        return None
    s = str(ts).strip()
    if not s:
        return None
    try:
        iso = s.replace("Z", "+00:00") if s.endswith("Z") else s
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _description_signature(text: str) -> str:
    raw = re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())
    words = [w for w in raw.split() if len(w) >= DESC_WORD_MIN_LEN]
    if len(words) < 2:
        return ""
    uniq = sorted(set(words))[:DESC_SIGNATURE_MAX_WORDS]
    return "|".join(uniq)


def _alert(
    *,
    type_: str,
    message: str,
    severity: str,
    count: int,
) -> dict[str, Any]:
    return {"type": type_, "message": message, "severity": severity, "count": int(count)}


def _severity_rank(s: str) -> int:
    return {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(s.upper(), 3)


def build_anomaly_alerts(vector_store: VectorStore) -> dict[str, Any]:
    rows = _fetch_all_claim_rows(vector_store)
    alerts: list[dict[str, Any]] = []

    total = len(rows)
    if total == 0:
        return {"alerts": []}

    decision_counts = {"APPROVED": 0, "REJECTED": 0, "INVESTIGATE": 0}
    hitl_n = 0
    high_fraud_by_product: dict[str, int] = defaultdict(int)
    product_counts: Counter[str] = Counter()
    desc_sig_counts: Counter[str] = Counter()
    rows_with_time: list[tuple[Optional[datetime], dict[str, Any], str]] = []

    for row in rows:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        d = _normalize_decision(meta)
        if d:
            decision_counts[d] += 1
        if _hitl_needed(meta):
            hitl_n += 1

        entities = _parse_entities_json(meta)
        pv = _scalar_entity_value(entities, "product") or _scalar_entity_value(entities, "product_code")
        if pv:
            product_counts[pv] += 1

        fs = _safe_fraud_score(meta)
        if fs is not None and fs > FRAUD_CLUSTER_SCORE_MIN and pv:
            high_fraud_by_product[pv] += 1

        doc = row.get("claim_description")
        doc_s = doc if isinstance(doc, str) else str(doc or "")
        sig = _description_signature(doc_s)
        if sig:
            desc_sig_counts[sig] += 1

        dt = _parse_meta_datetime(meta)
        rows_with_time.append((dt, meta, doc_s))

    dec_total = sum(decision_counts.values())
    if dec_total >= DECISION_RATIO_MIN_TOTAL:
        inv = decision_counts["INVESTIGATE"]
        if inv / dec_total > INVESTIGATE_RATIO_ALERT:
            alerts.append(
                _alert(
                    type_="SPIKE_IN_INVESTIGATIONS",
                    message="Unusual increase in INVESTIGATE decisions",
                    severity="MEDIUM",
                    count=inv,
                )
            )
        rej = decision_counts["REJECTED"]
        if rej / dec_total > REJECTION_RATIO_ALERT:
            alerts.append(
                _alert(
                    type_="SPIKE_IN_REJECTIONS",
                    message="Unusual increase in REJECTED decisions relative to all triaged claims",
                    severity="MEDIUM",
                    count=rej,
                )
            )

    for product, n in sorted(high_fraud_by_product.items(), key=lambda x: -x[1]):
        if n >= FRAUD_CLUSTER_MIN_CLAIMS:
            alerts.append(
                _alert(
                    type_="HIGH_FRAUD_CLUSTER",
                    message=f"Multiple high fraud claims detected for {product}",
                    severity="HIGH",
                    count=n,
                )
            )

    if total >= REPEAT_PRODUCT_MIN_TOTAL:
        for product, n in product_counts.most_common():
            if n >= REPEAT_PRODUCT_MIN_COUNT and n / total >= REPEAT_PRODUCT_SHARE:
                alerts.append(
                    _alert(
                        type_="REPEAT_PATTERN",
                        message=f"Repeated claims concentrated on product or entity: {product}",
                        severity="MEDIUM",
                        count=n,
                    )
                )

    top_sig = desc_sig_counts.most_common(1)
    if top_sig:
        sig, n = top_sig[0]
        if n >= REPEAT_DESC_MIN_SAME:
            preview = sig.replace("|", " ")[:72]
            suffix = "…" if len(sig.replace("|", " ")) > 72 else ""
            alerts.append(
                _alert(
                    type_="REPEAT_PATTERN",
                    message=f"Repeated claims with similar description pattern ({preview}{suffix})",
                    severity="MEDIUM",
                    count=n,
                )
            )

    if total >= HITL_MIN_TOTAL and hitl_n / total >= HITL_RATIO_ALERT:
        alerts.append(
            _alert(
                type_="HIGH_REVIEW_LOAD",
                message="Many claims require human review (HITL)",
                severity="LOW",
                count=hitl_n,
            )
        )

    timed = [(t, m, d) for t, m, d in rows_with_time if t is not None]
    if len(timed) >= 2 * TREND_WINDOW:
        timed.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc))
        recent_slice = timed[-TREND_WINDOW:]
        prev_slice = timed[-2 * TREND_WINDOW : -TREND_WINDOW]
        if len(prev_slice) >= TREND_MIN_PREV:
            r_inv = sum(1 for _, m, _ in recent_slice if _normalize_decision(m) == "INVESTIGATE")
            p_inv = sum(1 for _, m, _ in prev_slice if _normalize_decision(m) == "INVESTIGATE")
            r_ratio = r_inv / len(recent_slice)
            p_ratio = p_inv / len(prev_slice) if prev_slice else 0.0
            if r_ratio >= TREND_RECENT_RATIO and p_ratio > 0 and r_ratio >= TREND_SURGE_FACTOR * p_ratio:
                alerts.append(
                    _alert(
                        type_="RECENT_TREND_SURGE",
                        message=f"INVESTIGATE rate rose in the last {TREND_WINDOW} claims vs the prior {TREND_WINDOW}",
                        severity="MEDIUM",
                        count=r_inv,
                    )
                )

    alerts.sort(key=lambda a: (_severity_rank(str(a.get("severity", ""))), str(a.get("type", ""))))
    return {"alerts": alerts}


def _safe_confidence(meta: dict[str, Any]) -> Optional[float]:
    raw = meta.get("confidence")
    if raw is None or raw == "":
        return None
    try:
        x = float(raw)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _leaderboard_review_status(meta: dict[str, Any]) -> str:
    rs = str(meta.get("review_status") or "").strip().upper()
    if rs in REVIEW_KEYS:
        return rs
    ra = str(meta.get("reviewed_action") or "").strip().upper()
    if ra in REVIEW_KEYS:
        return ra
    return ""


def _human_review_settled(meta: dict[str, Any]) -> bool:
    return _leaderboard_review_status(meta) != ""


def _risk_level_label(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "HIGH"
    if risk_score >= 0.5:
        return "MEDIUM"
    return "LOW"


def risk_level_from_claim_metadata(meta: dict[str, Any]) -> str:
    """Same composite score as the fraud leaderboard (fraud + triage adjustments)."""
    fraud_raw = _safe_fraud_score(meta)
    fraud_base = fraud_raw if fraud_raw is not None else 0.0
    decision = _normalize_decision(meta)
    adj = 0.0
    if decision == "INVESTIGATE":
        adj += 0.1
    if not _human_review_settled(meta):
        adj += 0.05
    conf = _safe_confidence(meta)
    if conf is not None and conf < 0.6:
        adj += 0.05
    return _risk_level_label(fraud_base + adj)


def build_fraud_leaderboard(
    vector_store: VectorStore,
    *,
    limit: int = 10,
    min_fraud_score: Optional[float] = None,
) -> dict[str, Any]:
    """Rank claims by composite risk_score (fraud_score + triage/review adjustments)."""
    rows = _fetch_all_claim_rows(vector_store)
    scored: list[tuple[float, str, dict[str, Any]]] = []

    for row in rows:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        claim_id = str(row.get("claim_id") or "").strip()
        if not claim_id:
            continue

        fraud_raw = _safe_fraud_score(meta)
        fraud_base = fraud_raw if fraud_raw is not None else 0.0

        if min_fraud_score is not None and fraud_base < float(min_fraud_score):
            continue

        decision = _normalize_decision(meta)
        adj = 0.0
        if decision == "INVESTIGATE":
            adj += 0.1
        if not _human_review_settled(meta):
            adj += 0.05
        conf = _safe_confidence(meta)
        if conf is not None and conf < 0.6:
            adj += 0.05

        risk_score = fraud_base + adj
        scored.append((risk_score, claim_id, {"row": row, "meta": meta, "fraud_base": fraud_base, "fraud_raw": fraud_raw}))

    scored.sort(key=lambda t: (-t[0], t[1]))

    top: list[dict[str, Any]] = []
    cap = max(1, min(200, int(limit)))
    for risk_score, claim_id, bag in scored[:cap]:
        row = bag["row"]
        meta = bag["meta"]
        fraud_raw = bag["fraud_raw"]
        fraud_out = float(fraud_raw) if fraud_raw is not None else 0.0
        decision = _normalize_decision(meta)
        conf = _safe_confidence(meta)
        entities = _parse_entities_json(meta)
        doc = row.get("claim_description")
        desc = doc if isinstance(doc, str) else str(doc or "")
        ts = meta.get("timestamp")
        ts_out = str(ts).strip() if ts is not None else ""

        top.append(
            {
                "claim_id": claim_id,
                "fraud_score": round(fraud_out, 4),
                "decision": decision or "",
                "confidence": conf,
                "review_status": _leaderboard_review_status(meta),
                "description": desc,
                "entities": entities,
                "timestamp": ts_out,
                "risk_level": _risk_level_label(risk_score),
            }
        )

    return {"top_risky_claims": top}


def build_analytics_summary(vector_store: VectorStore) -> dict[str, Any]:
    rows = _fetch_all_claim_rows(vector_store)
    total = len(rows)

    decision_dist = {"APPROVED": 0, "REJECTED": 0, "INVESTIGATE": 0}
    review_dist = {"APPROVED": 0, "REJECTED": 0, "PENDING": 0}
    llm_used_count = 0
    llm_skipped_count = 0
    product_ctr: Counter[str] = Counter()
    band_ctr: Counter[str] = Counter()
    scores: list[float] = []
    day_ctr: Counter[str] = Counter()

    for row in rows:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        d = _normalize_decision(meta)
        if d:
            decision_dist[d] += 1

        r = _normalize_review_bucket(meta)
        review_dist[r] += 1

        fs = _safe_fraud_score(meta)
        if fs is not None:
            scores.append(fs)

        ds = str(meta.get("decision_source") or "").strip().lower()
        if ds == "rule":
            llm_skipped_count += 1
        elif ds == "llm":
            llm_used_count += 1
        elif ds == "fallback":
            # LLM was attempted (or analysis path triggered) but not reliable; still counts as "used"
            # for observability since it represents LLM pipeline invocation.
            llm_used_count += 1
        else:
            # Back-compat: treat missing as "unknown" and infer from llm_used when present.
            raw = str(meta.get("llm_used") or "").strip().lower()
            if raw in ("1", "true", "yes"):
                llm_used_count += 1
            elif raw in ("0", "false", "no"):
                llm_skipped_count += 1

        entities = _parse_entities_json(meta)
        pv = _scalar_entity_value(entities, "product") or _scalar_entity_value(entities, "product_code")
        if pv:
            product_ctr[pv] += 1
        bv = _scalar_entity_value(entities, "amount_band")
        if bv:
            band_ctr[bv] += 1

        day = _timestamp_to_day(meta)
        if day:
            day_ctr[day] += 1

    if scores:
        fraud_stats = {
            "avg": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
    else:
        fraud_stats = {"avg": 0.0, "min": 0.0, "max": 0.0}

    claims_over_time = [
        {"date": d, "count": int(day_ctr[d])}
        for d in sorted(day_ctr.keys())
    ]

    llm_usage_pct = round((llm_used_count / total * 100.0), 2) if total > 0 else 0.0

    return {
        "total_claims": total,
        "llm_used_count": int(llm_used_count),
        "llm_skipped_count": int(llm_skipped_count),
        "llm_usage_percentage": llm_usage_pct,
        "decision_distribution": decision_dist,
        "review_distribution": review_dist,
        "fraud_score_stats": fraud_stats,
        "top_entities": {
            "product": _top_kv(product_ctr),
            "amount_band": _top_kv(band_ctr),
        },
        "claims_over_time": claims_over_time,
    }
