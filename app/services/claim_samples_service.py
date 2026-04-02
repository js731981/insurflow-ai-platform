from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def _sample_claims_paths() -> list[Path]:
    """Candidate locations for sample claims.

    We intentionally keep sample fixtures under `app/data/...` so the project is self-contained.
    """

    app_root = Path(__file__).resolve().parents[1]  # .../app
    return [
        app_root / "data" / "claims" / "sample_claims.json",
    ]


@lru_cache(maxsize=1)
def load_sample_claims() -> list[dict[str, Any]]:
    """Load sample micro-insurance claims from `app/data/claims/sample_claims.json`.

    Cached for fast responses to `/claim/samples`.
    """

    last_exc: Exception | None = None
    payload: Any = None
    for claims_path in _sample_claims_paths():
        try:
            raw = claims_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            break
        except FileNotFoundError as exc:
            last_exc = exc

    if payload is None:
        logger.error("Sample claims file not found (tried multiple paths).")
        raise RuntimeError("Sample claims file missing") from last_exc

    if not isinstance(payload, list):
        raise RuntimeError("Sample claims JSON must be a list")

    # Ensure dict entries so API response is deterministic.
    result: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            result.append(item)
        else:
            logger.warning("Ignoring non-object item in sample claims: %r", item)
    return result

