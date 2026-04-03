import os

from dotenv import load_dotenv
from pydantic import BaseModel


# Load environment variables from a .env file if present.
load_dotenv()


class Settings(BaseModel):
    """Minimal runtime configuration.

    Keep this lightweight until you add more config sources.
    """

    app_name: str = os.getenv("APP_NAME", "Insurance AI Decision Platform")
    debug: bool = os.getenv("DEBUG", "false").strip().lower() == "true"
    # Confirmed local LLM default for MVP.
    model_name: str = os.getenv("MODEL_NAME", "phi3")

    # LLM routing/execution configuration
    # Confirmed local-only default for MVP (Ollama).
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    llm_fallback_providers: str = os.getenv("LLM_FALLBACK_PROVIDERS", "").strip().lower()
    # Total attempts per provider (initial + retries). "Retry up to 2 times" => 3 attempts.
    llm_retries: int = int(os.getenv("LLM_RETRIES", "3"))
    llm_base_delay_s: float = float(os.getenv("LLM_BASE_DELAY_S", "0.5"))
    llm_max_delay_s: float = float(os.getenv("LLM_MAX_DELAY_S", "5.0"))
    # Local Ollama (e.g. phi3) often needs >60s for long fraud prompts + JSON output on CPU.
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "180"))

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


settings = Settings()

