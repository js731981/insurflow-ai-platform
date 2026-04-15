from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMProviderError(Exception):
    """Raised when a provider fails to produce an LLM completion."""

    def __init__(
        self,
        *,
        provider: str,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        # Convenience attribute for logging/telemetry (Exception doesn't define `.message`).
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body


class LLMProvider(ABC):
    """Strategy interface for LLM execution providers."""

    @abstractmethod
    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

