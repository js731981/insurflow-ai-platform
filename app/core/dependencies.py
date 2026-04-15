import threading

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.hitl_service import HitlService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore

_vector_store_lock = threading.Lock()
_vector_store_singleton: VectorStore | None = None

_embedding_lock = threading.Lock()
_embedding_singleton: EmbeddingService | None = None

_llm_lock = threading.Lock()
_llm_singleton: LLMService | None = None


def get_llm_service() -> LLMService:
    """Process-wide LLM service (reuses HTTP clients / avoids reconstructing providers each request)."""
    global _llm_singleton
    if _llm_singleton is not None:
        return _llm_singleton
    with _llm_lock:
        if _llm_singleton is None:
            _llm_singleton = LLMService(model_name=settings.llm_model)
    return _llm_singleton


def get_embedding_service() -> EmbeddingService:
    global _embedding_singleton
    if _embedding_singleton is not None:
        return _embedding_singleton
    with _embedding_lock:
        if _embedding_singleton is None:
            _embedding_singleton = EmbeddingService(
                base_url=settings.ollama_base_url,
                model=settings.ollama_embedding_model,
                timeout_s=settings.embedding_timeout_s,
            )
    return _embedding_singleton


def get_vector_store() -> VectorStore:
    """One Chroma PersistentClient per process.

    Chroma shares internal state by persist path; concurrent first-time construction
    can race and yield RustBindingsAPI without initialized bindings. A singleton
    avoids that and matches Chroma's refcount model (see client.close()).
    """
    global _vector_store_singleton
    if _vector_store_singleton is not None:
        return _vector_store_singleton
    with _vector_store_lock:
        if _vector_store_singleton is None:
            _vector_store_singleton = VectorStore(
                persist_dir=settings.chroma_persist_dir,
                collection_name=settings.chroma_collection,
            )
    return _vector_store_singleton


def shutdown_vector_store() -> None:
    global _vector_store_singleton
    with _vector_store_lock:
        if _vector_store_singleton is not None:
            _vector_store_singleton.close()
            _vector_store_singleton = None


async def shutdown_llm_embedding_clients() -> None:
    """Close shared httpx sessions (Ollama LLM + embeddings)."""
    global _llm_singleton, _embedding_singleton
    llm: LLMService | None = None
    emb: EmbeddingService | None = None
    with _llm_lock:
        llm = _llm_singleton
        _llm_singleton = None
    with _embedding_lock:
        emb = _embedding_singleton
        _embedding_singleton = None
    if llm is not None:
        await llm.aclose()
    if emb is not None:
        await emb.aclose()


def get_hitl_service() -> HitlService:
    # Business rule: when APPROVED, only auto-approve if confidence >= 0.7.
    return HitlService(approve_confidence_threshold=0.7)
