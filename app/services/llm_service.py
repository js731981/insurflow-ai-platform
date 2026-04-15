import logging
from typing import Any, Dict, Optional

from app.core.config import settings
from app.services.llm.providers.openai_provider import OpenAIProvider
from app.services.llm.providers.openrouter_provider import OpenRouterProvider
from app.services.llm.providers.ollama_provider import OllamaProvider
from app.services.llm.providers.base import LLMProviderError
from app.services.llm.router import LLMCompletion, LLMRouter, RetryPolicy

logger = logging.getLogger(__name__)


class LLMService:
    """Executor for LLM calls.

    Per layering rules:
    - `app.agents` decide what to do (decision makers).
    - `app.services` execute the work (LLM, DB, external APIs).
    """

    def __init__(self, model_name: str):
        # Prefer settings-driven default; keep `model_name` arg as an override.
        self._desired_model_name = str(model_name or "").strip() or str(settings.llm_model or "").strip() or "phi3:mini"
        self._default_model_name = self._desired_model_name
        self._model_available: bool | None = None

        logger.info(
            "LLM config \u2192 model=%s, timeout=%ss, provider=%s",
            settings.llm_model,
            int(settings.llm_timeout_s) if float(settings.llm_timeout_s).is_integer() else settings.llm_timeout_s,
            settings.llm_provider,
        )

        fallback = [p.strip() for p in settings.llm_fallback_providers.split(",") if p.strip()]

        providers: Dict[str, Any] = {
            "ollama": OllamaProvider(
                base_url=settings.ollama_base_url,
                timeout_s=settings.llm_timeout_s,
            ),
        }
        if settings.openai_api_key:
            providers["openai"] = OpenAIProvider(
                api_key=settings.openai_api_key,
                timeout_s=settings.llm_timeout_s,
            )
        if settings.openrouter_api_key:
            providers["openrouter"] = OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                timeout_s=settings.llm_timeout_s,
            )

        self._router = LLMRouter(
            primary_provider=settings.llm_provider,
            fallback_providers=fallback,
            providers=providers,
            timeout_s=settings.llm_timeout_s,
            retry_policy=RetryPolicy(
                # Env var LLM_RETRIES represents "number of retries after the initial attempt".
                # Router policy expects total attempts.
                max_attempts=max(1, 1 + settings.llm_retries),
                base_delay_s=settings.llm_base_delay_s,
                max_delay_s=settings.llm_max_delay_s,
            ),
        )
        self._providers_for_cleanup = providers
        self._providers = providers

    @property
    def default_model(self) -> str:
        return self._default_model_name

    @property
    def model_available(self) -> bool | None:
        return self._model_available

    async def warmup(self) -> None:
        """Best-effort provider warmup (reduces cold-start latency)."""
        p = self._providers.get("ollama")
        warm = getattr(p, "warmup", None)
        if p is not None and settings.llm_provider == "ollama":
            # Startup validation: ensure configured model exists, otherwise fall back.
            list_models = getattr(p, "list_models", None)
            if callable(list_models):
                try:
                    available = await list_models()
                except Exception:
                    available = set()
                desired = self._desired_model_name

                # Ollama commonly reports names with tags (e.g. "phi3:mini") while users
                # may configure the base name (e.g. "phi3"). Treat these as compatible,
                # but NEVER rewrite the configured model name (config must remain authoritative).
                def _base(name: str) -> str:
                    return name.split(":", 1)[0].strip().lower()

                desired_norm = desired.strip().lower()
                desired_base = _base(desired_norm)

                available_list = sorted(available)
                available_norm = {m.strip().lower() for m in available_list}
                available_bases = {_base(m) for m in available_norm}

                exact_match = any(m.strip().lower() == desired_norm for m in available_list)
                base_match = desired_base in available_bases

                if available and (not exact_match) and (not base_match):
                    fallback_model = available_list[0]
                    logger.warning(
                        "ollama_model_not_found_falling_back",
                        extra={"desired_model": desired, "fallback_model": fallback_model, "available_count": len(available)},
                    )
                    self._default_model_name = fallback_model
                    self._model_available = False
                else:
                    # If we couldn't list models, treat as unknown; if listed and present, it's available.
                    if not available:
                        self._model_available = None
                    else:
                        self._model_available = True
                logger.info("LLM routing \u2192 model=%s", self._default_model_name)

        if callable(warm):
            try:
                await warm(model=self._default_model_name)
            except Exception:
                # Warmup is optional; do not fail startup.
                return

    async def aclose(self) -> None:
        for p in self._providers_for_cleanup.values():
            aclose = getattr(p, "aclose", None)
            if callable(aclose):
                await aclose()

    async def generate(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        context: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        claim_id: Optional[str] = None,
        timeout_s: Optional[float] = None,
        max_attempts: Optional[int] = None,
    ) -> LLMCompletion:
        # Production note: keep prompt construction here so all agents remain transport-agnostic.
        model_to_use = model or self._default_model_name
        if isinstance(model_to_use, str) and model_to_use.strip().lower() == "default":
            model_to_use = self._default_model_name
        if context:
            full_prompt = f"{prompt}\n\nContext:\n{context}"
        else:
            full_prompt = prompt

        try:
            return await self._router.complete(
                prompt=full_prompt,
                model=model_to_use,
                generation_kwargs=generation_kwargs,
                provider=provider,
                claim_id=claim_id,
                timeout_s=timeout_s,
                max_attempts=max_attempts,
            )
        except LLMProviderError as exc:
            if exc.status_code == 404 and "model" in (exc.response_body or "").lower() and "not found" in (
                exc.response_body or ""
            ).lower():
                logger.warning("Model not available, falling back to safe decision")
            raise

