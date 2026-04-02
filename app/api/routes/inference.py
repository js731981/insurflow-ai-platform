import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.agents.orchestrator import InsurFlowOrchestrator, get_insurflow_orchestrator
from app.models.schemas import ClaimProcessResponse, ClaimRequest, InferenceRequest, InferenceResponse
from app.services.claim_samples_service import load_sample_claims

router = APIRouter()


@router.post("/inference", response_model=InferenceResponse, tags=["inference"])
async def inference(
    request: InferenceRequest,
    orchestrator: InsurFlowOrchestrator = Depends(get_insurflow_orchestrator),
) -> InferenceResponse:
    return await orchestrator.run_inference(request)


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
    body: ClaimRequest,
    orchestrator: InsurFlowOrchestrator = Depends(get_insurflow_orchestrator),
) -> ClaimProcessResponse:
    result = await orchestrator.process_claim(body.model_dump(exclude_none=True))
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
