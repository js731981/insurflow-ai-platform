from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Must be set before chromadb import (Chroma reads env when telemetry initializes).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

REVIEW_DECISIONS = frozenset({"APPROVED", "REJECTED"})


@dataclass(frozen=True)
class SimilarHit:
    claim_id: str
    distance: float
    document: str
    metadata: dict[str, Any]
    base_score: float
    adjusted_score: float


def _distance_to_base_score(distance: float) -> float:
    """Higher is better; Chroma returns lower distance for closer vectors."""
    d = float(distance)
    if d < 0:
        d = 0.0
    return 1.0 / (1.0 + d)


def _majority_review_decision(metas: list[dict[str, Any]]) -> Optional[str]:
    votes: list[str] = []
    for m in metas:
        rs = str(m.get("review_status") or "").strip().upper()
        if rs in REVIEW_DECISIONS:
            votes.append(rs)
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


def majority_review_from_similar_hits(similar_hits: list[SimilarHit]) -> Optional[str]:
    """Human review majority (APPROVED/REJECTED) among similar retrieved claims, or None."""
    return _majority_review_decision([h.metadata for h in similar_hits])


def format_similar_hits_for_context(hits: list[SimilarHit]) -> str:
    """Build LLM context block from weighted similar hits."""
    if not hits:
        return ""
    lines: list[str] = ["Similar claims from memory (weighted ranking):"]
    for idx, h in enumerate(hits[:6], start=1):
        preview = (h.document or "").strip().replace("\n", " ")[:220]
        decision = str(h.metadata.get("decision") or "")
        hum = _human_review_label(h.metadata)
        lines.append(f"{idx}. claim_id={h.claim_id} (score={h.adjusted_score:.3f})")
        lines.append(f"   Description: {preview}")
        lines.append(f"   Decision: {decision}")
        if hum:
            lines.append(f"   Human review: {hum}")
    return "\n".join(lines).strip()


def compute_calibrated_confidence(
    *,
    confidence: float,
    model_decision: str,
    similar_hits: list[SimilarHit],
) -> float:
    """Adjust confidence using majority human review among similar claims (with review_status)."""
    majority = _majority_review_decision([h.metadata for h in similar_hits])
    if majority is None:
        return max(0.0, min(1.0, float(confidence)))

    md = (model_decision or "").strip().upper()
    c = float(confidence)

    # Majority APPROVED: boost agreement, lean toward approval (softer penalty on INVESTIGATE).
    if majority == "APPROVED":
        if md == "APPROVED":
            out = c * 1.18
        elif md == "INVESTIGATE":
            out = c * 0.93
        else:
            out = c * 0.65
        return max(0.0, min(1.0, out))

    if majority == "REJECTED":
        if md == "REJECTED":
            out = c * 1.1
        elif md == "APPROVED":
            out = c * 0.7
        else:
            out = c * 0.75
        return max(0.0, min(1.0, out))

    return max(0.0, min(1.0, c))  # defensive (majority is only APPROVED|REJECTED here)


class VectorStore:
    """Minimal ChromaDB wrapper (embedded, persistent)."""

    def __init__(
        self,
        *,
        persist_dir: str,
        collection_name: str = "claims",
    ) -> None:
        # Absolute path stabilizes Chroma's singleton key (same dir as ./chroma_db vs cwd).
        self._persist_dir = str(Path(persist_dir).expanduser().resolve())
        self._collection_name = collection_name
        # Disable Chroma PostHog telemetry (avoids noisy errors + fits local-only MVP).
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def close(self) -> None:
        """Release Chroma client refcount (important when using a process-wide singleton)."""
        if getattr(self, "_client", None) is None:
            return
        try:
            self._client.close()
        except Exception:
            logger.exception("vector_store_close_failed")
        self._client = None  # type: ignore[assignment]

    def store_claim(
        self,
        *,
        claim_id: str,
        claim_description: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        claim_id_n = (claim_id or "").strip()
        if not claim_id_n:
            raise ValueError("claim_id is required for vector store store_claim().")

        _validate_embedding(embedding)

        meta = _normalize_metadata_for_chroma(dict(metadata or {}))
        _validate_claim_metadata(meta, claim_id=claim_id_n)

        self._collection.upsert(
            ids=[claim_id_n],
            documents=[claim_description or ""],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def query_similar_hits(
        self,
        *,
        query_embedding: list[float],
        exclude_claim_id: Optional[str] = None,
        n_results: int = 10,
    ) -> list[SimilarHit]:
        """Retrieve similar claims with weighted ranking (review + decision coherence)."""
        _validate_embedding(query_embedding)

        n = max(1, min(25, n_results))
        try:
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            logger.exception("vector_query_failed")
            return []

        ids_batch = _chroma_query_inner_list(res, "ids")
        docs_batch = _chroma_query_inner_list(res, "documents")
        metas_batch = _chroma_query_inner_list(res, "metadatas")
        dist_batch = _chroma_query_inner_list(res, "distances")

        exclude = (exclude_claim_id or "").strip()
        raw: list[tuple[str, float, str, dict[str, Any]]] = []
        for i, cid in enumerate(ids_batch):
            if exclude and str(cid) == exclude:
                continue
            dist = float(dist_batch[i]) if i < len(dist_batch) else 0.0
            d = docs_batch[i] if i < len(docs_batch) else ""
            doc = d if isinstance(d, str) else str(d or "")
            m = metas_batch[i] if i < len(metas_batch) else {}
            meta = m if isinstance(m, dict) else {}
            raw.append((str(cid), dist, doc, meta))

        if not raw:
            return []

        majority_review = _majority_review_decision([m for _, _, _, m in raw])

        hits: list[SimilarHit] = []
        for cid, dist, doc, meta in raw:
            base = _distance_to_base_score(dist)
            boost_review = 0.1 if _is_reviewed(meta) else 0.0
            stored_decision = str(meta.get("decision") or "").strip().upper()
            boost_match = 0.0
            if majority_review and stored_decision == majority_review:
                boost_match = 0.05
            adjusted = base + boost_review + boost_match
            hits.append(
                SimilarHit(
                    claim_id=cid,
                    distance=dist,
                    document=doc,
                    metadata=meta,
                    base_score=base,
                    adjusted_score=adjusted,
                )
            )

        hits.sort(key=lambda h: h.adjusted_score, reverse=True)
        return hits

    def query_similar_for_context(
        self,
        *,
        query_embedding: list[float],
        exclude_claim_id: Optional[str] = None,
        n_results: int = 8,
    ) -> str:
        """Return a short text block prioritising weighted similar claims."""
        hits = self.query_similar_hits(
            query_embedding=query_embedding,
            exclude_claim_id=exclude_claim_id,
            n_results=n_results,
        )
        return format_similar_hits_for_context(hits)

    def get_claim(self, claim_id: str) -> Optional[dict[str, Any]]:
        claim_id_n = (claim_id or "").strip()
        if not claim_id_n:
            return None

        res = self._collection.get(
            ids=[claim_id_n],
            include=["documents", "metadatas", "embeddings"],
        )
        ids = res.get("ids") if res else None
        if ids is None or _seq_len(ids) == 0:
            return None

        doc, meta, emb = _first_row_from_chroma_get(res)
        return {
            "claim_id": claim_id_n,
            "claim_description": doc,
            "metadata": meta,
            "embedding": emb,
        }

    def list_claims(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        res = self._collection.get(
            include=["documents", "metadatas"],
            limit=max(1, min(500, limit)),
            offset=max(0, offset),
        )

        ids = _seq_to_list(res.get("ids"))
        docs = _seq_to_list(res.get("documents"))
        metas = _seq_to_list(res.get("metadatas"))

        out: list[dict[str, Any]] = []
        for i, claim_id in enumerate(ids):
            d = docs[i] if i < len(docs) else ""
            doc_str = d if isinstance(d, str) else str(d or "")
            m = metas[i] if i < len(metas) else {}
            meta_d = m if isinstance(m, dict) else {}
            out.append(
                {
                    "claim_id": claim_id,
                    "claim_description": doc_str,
                    "metadata": meta_d,
                }
            )
        return out

    def count_stored_claims(self) -> int:
        """Rows in the persistent collection (survives API restarts; not the same as in-memory triage counter)."""
        return int(self._collection.count())


def _seq_to_list(x: Any) -> list[Any]:
    if x is None:
        return []
    try:
        return list(x)
    except TypeError:
        return []


def _chroma_query_inner_list(res: dict[str, Any], key: str) -> list[Any]:
    """First query batch row for key (Chroma returns nested lists; inner may be numpy)."""
    outer = res.get(key)
    if outer is None or _seq_len(outer) == 0:
        return []
    inner = outer[0]
    if inner is None:
        return []
    return _seq_to_list(inner)


def _seq_len(x: Any) -> int:
    try:
        return len(x)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _first_row_from_chroma_get(res: dict[str, Any]) -> tuple[str, dict[str, Any], list[float]]:
    """Extract first document, metadata, embedding (Chroma may return numpy arrays — avoid `or` / bool on them)."""
    docs = res.get("documents")
    doc = ""
    if docs is not None and _seq_len(docs) > 0:
        d0 = docs[0]
        doc = d0 if isinstance(d0, str) else str(d0 or "")

    metas = res.get("metadatas")
    meta: dict[str, Any] = {}
    if metas is not None and _seq_len(metas) > 0:
        m0 = metas[0]
        if isinstance(m0, dict):
            meta = m0

    emb: list[float] = []
    embs = res.get("embeddings")
    if embs is not None and _seq_len(embs) > 0:
        first = embs[0]
        if hasattr(first, "tolist"):
            emb = [float(x) for x in first.tolist()]
        elif isinstance(first, (list, tuple)):
            emb = [float(x) for x in first]

    return doc, meta, emb


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_reviewed(meta: dict[str, Any]) -> bool:
    return bool(
        str(meta.get("review_status") or "").strip()
        or str(meta.get("reviewed_action") or "").strip()
    )


def _human_review_label(meta: dict[str, Any]) -> str:
    rs = str(meta.get("review_status") or "").strip()
    if rs:
        return rs
    ra = str(meta.get("reviewed_action") or "").strip()
    return ra


def _normalize_metadata_for_chroma(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be scalar; serialize entities as JSON string."""
    out = dict(meta)
    ent = out.get("entities")
    if isinstance(ent, (dict, list)):
        out["entities_json"] = json.dumps(ent, ensure_ascii=False)
        del out["entities"]
    elif "entities_json" not in out:
        out["entities_json"] = "{}"

    for k, v in list(out.items()):
        if v is None:
            out[k] = ""
    return out


def _validate_claim_metadata(meta: dict[str, Any], *, claim_id: str) -> None:
    """
    Enforce the MVP contract: each stored record contains embedding + document + full metadata.
    """
    required = (
        "claim_id",
        "fraud_score",
        "decision",
        "confidence",
        "entities_json",
        "timestamp",
        "explanation",
        "review_status",
        "case_status",
        "assigned_to",
        "assigned_at",
        "updated_at",
    )
    meta.setdefault("claim_id", claim_id)
    meta.setdefault("timestamp", _utc_now_iso())
    meta.setdefault("explanation", "")
    meta.setdefault("review_status", "")
    meta.setdefault("case_status", "NEW")
    meta.setdefault("assigned_to", "")
    meta.setdefault("assigned_at", "")
    if not str(meta.get("updated_at") or "").strip():
        meta["updated_at"] = str(meta.get("timestamp") or _utc_now_iso())

    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"Missing required metadata fields: {missing}")

    expl = str(meta.get("explanation") or "").strip()
    if not expl:
        raise ValueError("explanation must be non-empty for stored claims.")


def _validate_embedding(embedding: Any) -> None:
    if embedding is None:
        raise ValueError("Embedding is required for storing claim (got None).")
    if not isinstance(embedding, list):
        raise ValueError(f"Embedding must be a list[float] (got {type(embedding).__name__}).")
    if len(embedding) == 0:
        raise ValueError("Embedding is required for storing claim (got empty list).")
    if len(embedding) < 10:
        raise ValueError(f"Invalid embedding size: {len(embedding)} (expected >= 10).")
