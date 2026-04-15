import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load environment variables from a .env file if present.
load_dotenv()


def _default_llm_timeout_seconds() -> float:
    """Wall-clock cap per LLM attempt.

    The UI pipeline must remain responsive; prefer a strict default and allow override via env.
    """
    prov = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    raw = os.getenv("LLM_TIMEOUT_S") or os.getenv("LLM_TIMEOUT")
    if raw is not None and str(raw).strip() != "":
        try:
            # Safety floor: Ollama cold starts and longer prompts can exceed 20s.
            # Enforce >=60s even if env is set lower.
            return max(60.0, float(raw))
        except (TypeError, ValueError):
            return 60.0
    # Default: production-friendly cap per attempt.
    # Ollama + local models can take >10s on cold start or long prompts.
    _ = prov  # keep provider context for future tuning
    return 60.0


def _default_claim_timeout_seconds() -> float:
    """Overall claim pipeline target for interactive UI.

    The system should return a result quickly; if the LLM is slow/unavailable,
    we fall back to a safe INVESTIGATE decision.
    """
    raw = os.getenv("CLAIM_TIMEOUT_S") or os.getenv("CLAIM_TIMEOUT")
    if raw is not None and str(raw).strip() != "":
        return float(raw)
    return 35.0


def _default_fraud_agent_timeout_seconds() -> float:
    """Hard cap for fraud agent LLM work inside a claim request."""
    raw = os.getenv("FRAUD_AGENT_TIMEOUT_S") or os.getenv("FRAUD_AGENT_TIMEOUT")
    if raw is not None and str(raw).strip() != "":
        return float(raw)
    # Separate from LLM timeout. This is an overall "agent budget" (prompt build + LLM + parsing).
    # Keep comfortably above the per-attempt LLM timeout to avoid false timeouts after the LLM returns.
    return 120.0


class Settings(BaseModel):
    """Minimal runtime configuration.

    Keep this lightweight until you add more config sources.
    """

    app_name: str = os.getenv("APP_NAME", "Insurance AI Decision Platform")
    debug: bool = os.getenv("DEBUG", "false").strip().lower() == "true"
    # Primary chat/completion model name.
    #
    # New config key: LLM_MODEL (preferred).
    # Back-compat alias: MODEL_NAME (deprecated).
    llm_model: str = os.getenv("LLM_MODEL", "").strip() or os.getenv("MODEL_NAME", "phi3:mini").strip()
    # Back-compat attribute used across the codebase today.
    model_name: str = llm_model

    # LLM routing/execution configuration
    # Confirmed local-only default for MVP (Ollama).
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    llm_fallback_providers: str = os.getenv("LLM_FALLBACK_PROVIDERS", "").strip().lower()
    # Total attempts per provider (initial + retries). "Retry up to 2 times" => 3 attempts.
    llm_retries: int = int(os.getenv("LLM_RETRIES", "2"))
    llm_base_delay_s: float = float(os.getenv("LLM_BASE_DELAY_S", "0.5"))
    llm_max_delay_s: float = float(os.getenv("LLM_MAX_DELAY_S", "5.0"))
    # Per-attempt wall-clock cap (asyncio.wait_for + provider HTTP client). Alias: LLM_TIMEOUT.
    llm_timeout_s: float = Field(default_factory=_default_llm_timeout_seconds)
    # Interactive API cap: entire claim pipeline wall-clock budget.
    claim_timeout_s: float = Field(default_factory=_default_claim_timeout_seconds)
    # Hard cap for FraudAgent run within claim processing.
    fraud_agent_timeout_s: float = Field(default_factory=_default_fraud_agent_timeout_seconds)

    # Estimated cost tracking (heuristic; providers do not currently return usage).
    # Provide USD rates per 1k tokens. If zero, cost will generally compute as 0.
    llm_cost_usd_per_1k_input_tokens: float = float(os.getenv("LLM_COST_USD_PER_1K_INPUT_TOKENS", "0"))
    llm_cost_usd_per_1k_output_tokens: float = float(os.getenv("LLM_COST_USD_PER_1K_OUTPUT_TOKENS", "0"))

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Embeddings + local vector store (fully embedded / local persistence)
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "claims")

    # RAG (retrieval-augmented fraud context). TOP_K = RAG_TOP_K (similar claims count).
    rag_enabled: bool = os.getenv("RAG_ENABLED", "true").strip().lower() == "true"
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "1"))
    rag_rerank_enabled: bool = os.getenv("RAG_RERANK_ENABLED", "false").strip().lower() == "true"
    rag_context_max_tokens: int = int(os.getenv("RAG_CONTEXT_MAX_TOKENS", "256"))

    enable_parallel_execution: bool = (
        os.getenv("ENABLE_PARALLEL_EXECUTION", "true").strip().lower() == "true"
    )

    embedding_timeout_s: float = float(os.getenv("EMBEDDING_TIMEOUT_S", "30"))

    # Lightweight DL fraud head (optional torch; logistic fallback without it)
    dl_fraud_enabled: bool = os.getenv("DL_FRAUD_ENABLED", "false").strip().lower() == "true"
    dl_fraud_fusion_llm_weight: float = float(os.getenv("DL_FRAUD_FUSION_LLM_WEIGHT", "0.7"))
    dl_fraud_fusion_dl_weight: float = float(os.getenv("DL_FRAUD_FUSION_DL_WEIGHT", "0.3"))

    # FraudAgent: JSON parse recovery (extra LLM calls after invalid structured output).
    max_llm_retries: int = int(os.getenv("MAX_LLM_RETRIES", "0"))
    strict_json_mode: bool = os.getenv("STRICT_JSON_MODE", "true").strip().lower() == "true"

    # Multimodal image triage (CPU-friendly heuristics; optional CNN feature backend).
    enable_image_analysis: bool = os.getenv("ENABLE_IMAGE_ANALYSIS", "true").strip().lower() == "true"
    image_model_type: str = os.getenv("IMAGE_MODEL_TYPE", "heuristic").strip().lower()
    image_fusion_weight: float = float(os.getenv("IMAGE_FUSION_WEIGHT", "0.2"))
    # CNN classification gating: only trust CNN label when confidence is high.
    cnn_conf_threshold: float = float(os.getenv("CNN_CONF_THRESHOLD", "0.7"))


settings = Settings()

# Simple, module-level config values for codepaths that prefer direct imports.
# These are sourced from `settings`, which already supports `.env` overrides.
LLM_MODEL: str = settings.llm_model
LLM_TIMEOUT: float = float(settings.llm_timeout_s)  # default 60s; overridable via env LLM_TIMEOUT / LLM_TIMEOUT_S
AGENT_TIMEOUT: float = float(settings.fraud_agent_timeout_s)  # default 120s; overridable via env FRAUD_AGENT_TIMEOUT(_S)
RAG_TOP_K: int = int(settings.rag_top_k)

# CNN confidence threshold (0..1). Used to gate CNN classification outputs.
CNN_CONF_THRESHOLD: float = float(settings.cnn_conf_threshold)

