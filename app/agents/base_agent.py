from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


def _preview_dict(data: dict[str, Any], max_chars: int = 2000) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, default=str)
    except TypeError:
        text = str(data)
    if len(text) > max_chars:
        return f"{text[:max_chars]}...<truncated>"
    return text


class BaseAgent(ABC):
    """Async agent contract for micro-insurance workflows (timing + structured logging)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Agent-specific logic; invoked by `run` after logging starts."""

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        logger.info(
            "agent_run_start",
            extra={
                "agent": self.name,
                "input": _preview_dict(input_data),
            },
        )
        try:
            result = await self._execute(input_data)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "agent_run_failed",
                extra={"agent": self.name, "duration_ms": round(elapsed_ms, 2)},
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "agent_run_complete",
            extra={
                "agent": self.name,
                "duration_ms": round(elapsed_ms, 2),
                "output": _preview_dict(result),
            },
        )
        return result
