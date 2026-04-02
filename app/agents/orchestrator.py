from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import Depends

from app.agents.decision_agent import DecisionAgent
from app.agents.fraud_agent import FraudAgent
from app.agents.policy_agent import PolicyAgent
from app.core.dependencies import get_llm_service
from app.models.schemas import InferenceRequest, InferenceResponse
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class InsurFlowOrchestrator:
    """Micro-insurance claim pipeline: parallel fraud + policy checks, then a decision."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm_service = llm_service
        self._fraud_agent = FraudAgent(llm_service=llm_service)
        self._policy_agent = PolicyAgent()
        self._decision_agent = DecisionAgent()

    async def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        provider_override: str | None = None
        if request.task:
            task_l = request.task.strip().lower()
            if task_l == "cheap":
                provider_override = "ollama"
            elif task_l == "complex":
                provider_override = "openai"

        completion = await self._llm_service.generate(
            prompt=request.prompt,
            context=request.context,
            model=request.model,
            provider=provider_override,
        )
        return InferenceResponse(
            text=completion.text,
            provider=completion.provider,
            model=completion.model,
            tokens=completion.tokens,
            cost=completion.cost,
            latency=completion.latency_ms,
            confidence=completion.confidence,
        )

    async def process_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Run fraud and policy agents in parallel, then aggregate for auto-triage."""
        workflow_start = time.perf_counter()
        logger.info(
            "orchestrator_claim_start",
            extra={"claim_keys": list(claim.keys())},
        )

        fraud_task = self._fraud_agent.run(claim)
        policy_task = self._policy_agent.run(claim)
        fraud_out, policy_out = await asyncio.gather(fraud_task, policy_task)

        decision_in: dict[str, Any] = {"fraud": fraud_out, "policy": policy_out}
        decision_out = await self._decision_agent.run(decision_in)

        elapsed_ms = (time.perf_counter() - workflow_start) * 1000
        logger.info(
            "orchestrator_claim_complete",
            extra={
                "duration_ms": round(elapsed_ms, 2),
                "decision": decision_out.get("decision"),
            },
        )

        claim_id = str(claim.get("claim_id") or "unknown")
        return {
            "claim_id": claim_id,
            "decision": decision_out["decision"],
            "confidence_score": decision_out["confidence_score"],
            "agent_outputs": {
                "fraud": fraud_out,
                "policy": policy_out,
                "decision": decision_out,
            },
        }


def get_insurflow_orchestrator(
    llm_service: LLMService = Depends(get_llm_service),
) -> InsurFlowOrchestrator:
    return InsurFlowOrchestrator(llm_service=llm_service)
