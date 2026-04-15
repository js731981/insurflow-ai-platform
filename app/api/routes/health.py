from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.dependencies import get_llm_service, get_vector_store
from app.services.llm_service import LLMService
from app.services.metrics import metrics
from app.services.vector_store import VectorStore

router = APIRouter()


@router.get("/", tags=["health"])
def root() -> dict:
    return {
        "name": settings.app_name,
        "status": "ok",
        "docs": "/docs",
        "ui": "/ui",
    }


@router.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


@router.get("/metrics", tags=["health"])
def get_metrics(vector_store: VectorStore = Depends(get_vector_store)) -> dict:
    """In-memory counters reset on every API restart; see `metrics_scope` and `vector_store_claim_documents`."""
    out: dict = dict(metrics.snapshot())
    out["metrics_scope"] = "since_api_process_start_counters_reset_on_restart"
    try:
        out["vector_store_claim_documents"] = vector_store.count_stored_claims()
    except Exception:
        out["vector_store_claim_documents"] = None
    return out


@router.get("/health/llm", tags=["health"])
async def llm_health(llm: LLMService = Depends(get_llm_service)) -> dict:
    # Ensure warmup/model validation has had a chance to run.
    try:
        await llm.warmup()
    except Exception:
        pass
    return {
        "provider": settings.llm_provider,
        "model_configured": settings.llm_model,
        "model_in_use": llm.default_model,
        "model_available": llm.model_available,
    }

