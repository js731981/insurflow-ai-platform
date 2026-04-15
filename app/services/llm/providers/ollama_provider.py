from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Set

import httpx

from app.services.llm.providers.base import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama `/api/generate` using a shared async HTTP client (no per-request model reload on client side)."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._provider_name = "ollama"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_s),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def list_models(self) -> Set[str]:
        """Best-effort list of local model names available in Ollama."""
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - best-effort; caller decides fallback
            logger.warning("ollama_list_models_failed", extra={"error": f"{type(exc).__name__}: {exc}"})
            return set()

        models: set[str] = set()
        items = data.get("models") if isinstance(data, dict) else None
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    name = it.get("name")
                    if isinstance(name, str) and name.strip():
                        models.add(name.strip())
        return models

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # noqa: BLE001
            logger.exception("ollama_client_close_failed")

    async def warmup(self, *, model: str) -> None:
        """Best-effort warmup to reduce first-request cold start latency."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": "ping",
            "stream": False,
            "options": {"num_predict": 1},
        }
        try:
            resp = await self._client.post("/api/generate", json=payload)
            if resp.status_code == 404:
                # Some deployments expose chat but not generate; try chat warmup too.
                resp = await self._client.post(
                    "/api/chat",
                    json={"model": model, "messages": [{"role": "user", "content": "ping"}], "stream": False},
                )
            resp.raise_for_status()
            logger.info("ollama_warmup_ok", extra={"model": model})
        except Exception as exc:  # noqa: BLE001
            logger.warning("ollama_warmup_failed", extra={"model": model, "error": f"{type(exc).__name__}: {exc}"})

    async def _complete_chat(self, *, prompt: str, model: str, generation_kwargs: Optional[Dict[str, Any]]) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if generation_kwargs:
            # Ollama chat supports a subset of options; pass through best-effort.
            payload.update(generation_kwargs)

        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            body = ""
            try:
                body = exc.response.text[:2000]
            except Exception:
                body = str(exc)
            if exc.response.status_code == 404 and "model" in body.lower() and "not found" in body.lower():
                raise LLMProviderError(
                    provider=self._provider_name,
                    message="Model not available in Ollama (404)",
                    status_code=exc.response.status_code,
                    response_body=body,
                ) from exc
            raise LLMProviderError(
                provider=self._provider_name,
                message=f"Ollama HTTP error ({exc.response.status_code})",
                status_code=exc.response.status_code,
                response_body=body,
            ) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(
                provider=self._provider_name,
                message=f"Ollama network error: {exc}",
            ) from exc

        msg = data.get("message") if isinstance(data, dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str):
            raise LLMProviderError(
                provider=self._provider_name,
                message="Unexpected Ollama chat response format",
                response_body=json.dumps(data)[:2000] if isinstance(data, dict) else str(data)[:2000],
            )
        return content

    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        gen = dict(generation_kwargs or {})

        # Required request shape (and sane defaults) for reliable Ollama latency.
        # Keep per-call overrides working via generation_kwargs.
        opts: Dict[str, Any] = {
            "temperature": 0.2,
            "num_predict": 200,
        }

        # Provider-normalization: accept common kwargs and map to Ollama options.
        # - `max_tokens` (OpenAI-style) => `options.num_predict` (Ollama)
        # - `temperature` => `options.temperature`
        if isinstance(gen.get("options"), dict):
            opts.update(gen.get("options") or {})

        if "max_tokens" in gen:
            try:
                mt = int(gen.get("max_tokens") or 0)
            except (TypeError, ValueError):
                mt = 0
            if mt > 0:
                opts["num_predict"] = mt
            gen.pop("max_tokens", None)

        if "temperature" in gen:
            try:
                temp = float(gen.get("temperature"))
            except (TypeError, ValueError):
                temp = None
            if temp is not None:
                opts["temperature"] = temp
            gen.pop("temperature", None)

        # Remove any top-level keys that would conflict with the required payload shape.
        gen.pop("model", None)
        gen.pop("prompt", None)
        gen.pop("stream", None)

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": opts,
        }
        # Allow any additional Ollama fields (e.g. `format`, `keep_alive`) through.
        # If caller provided its own `options`, we already merged it into `opts`.
        gen.pop("options", None)
        if gen:
            payload.update(gen)

        try:
            resp = await self._client.post("/api/generate", json=payload)
            if resp.status_code == 404:
                # Fallback: some Ollama-compatible servers only expose chat.
                return await self._complete_chat(prompt=prompt, model=model, generation_kwargs=generation_kwargs)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            body = ""
            try:
                body = exc.response.text[:2000]
            except Exception:
                body = str(exc)
            if exc.response.status_code == 404:
                # Second chance: try chat if generate isn't supported.
                return await self._complete_chat(prompt=prompt, model=model, generation_kwargs=generation_kwargs)
            raise LLMProviderError(
                provider=self._provider_name,
                message=f"Ollama HTTP error ({exc.response.status_code})",
                status_code=exc.response.status_code,
                response_body=body,
            ) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(
                provider=self._provider_name,
                message=f"Ollama network error: {exc}",
            ) from exc

        response = data.get("response")
        if not isinstance(response, str):
            raise LLMProviderError(
                provider=self._provider_name,
                message="Unexpected Ollama response format",
                response_body=json.dumps(data)[:2000],
            )
        return response
