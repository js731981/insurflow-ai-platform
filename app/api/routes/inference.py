import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from app.agents.orchestrator import InsurFlowOrchestrator, get_insurflow_orchestrator
from app.api.claim_multipart import parse_claim_http_request
from app.models.schemas import ClaimProcessResponse, ClaimRequest, InferenceRequest, InferenceResponse
from app.services.llm.providers.base import LLMProviderError
from app.services.claim_samples_service import load_sample_claims

router = APIRouter()


@router.post("/inference", response_model=InferenceResponse, tags=["inference"])
async def inference(
    request: InferenceRequest,
    orchestrator: InsurFlowOrchestrator = Depends(get_insurflow_orchestrator),
) -> InferenceResponse:
    try:
        return await orchestrator.run_inference(request)
    except LLMProviderError as exc:
        # Do not 500 on local model not installed; return a safe, explicit fallback payload.
        if exc.status_code == 404:
            return InferenceResponse(
                text="Model not available, falling back to safe decision",
                provider=exc.provider,
                model=str(request.model or "default"),
                tokens=0,
                cost=0.0,
                latency=0,
                confidence=0.0,
            )
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc.message}") from exc
    except Exception as exc:  # noqa: BLE001 - safe API error
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}") from exc


@router.post(
    "/claim",
    response_model=ClaimProcessResponse,
    tags=["micro-insurance"],
    summary="Triage a micro-insurance claim",
    description=(
        "Runs fraud and policy checks in parallel, then returns APPROVED, REJECTED, or INVESTIGATE."
    ),
)
async def process_claim(
    request: Request,
    orchestrator: InsurFlowOrchestrator = Depends(get_insurflow_orchestrator),
) -> ClaimProcessResponse:
    payload = await parse_claim_http_request(request)
    result = await orchestrator.process_claim(payload)
    return ClaimProcessResponse.model_validate(result)


@router.get(
    "/claim/samples",
    response_model=list[ClaimRequest],
    tags=["micro-insurance"],
    summary="Get sample micro-insurance claims",
)
async def claim_samples() -> list[ClaimRequest]:
    try:
        payload = await asyncio.to_thread(load_sample_claims)
    except Exception as exc:  # noqa: BLE001 - safe API error
        raise HTTPException(status_code=500, detail=f"Failed to load sample claims: {type(exc).__name__}") from exc
    return [ClaimRequest.model_validate(item) for item in payload]
