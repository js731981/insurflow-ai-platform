import threading

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.hitl_service import HitlService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore

_vector_store_lock = threading.Lock()
_vector_store_singleton: VectorStore | None = None


def get_llm_service() -> LLMService:
    # Factory so each request gets its own service instance (swap later if you want pooling).
    return LLMService(model_name=settings.model_name)


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


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


def get_hitl_service() -> HitlService:
    return HitlService(confidence_threshold=0.75)

