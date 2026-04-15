import base64
import binascii
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    context: Optional[str] = Field(default=None, max_length=20000)
    model: str = Field(default="default", max_length=100)
    task: Optional[str] = Field(default=None, max_length=50)


class InferenceResponse(BaseModel):
    text: str
    provider: str
    model: str
    tokens: int
    cost: float
    latency: int
    confidence: float = Field(default=0.9)


class FraudAnalysisResponse(BaseModel):
    """Legacy fraud-only shape (optional use by clients)."""

    fraud_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=2000)


ClaimDecision = Literal["APPROVED", "REJECTED", "INVESTIGATE"]
DecisionSource = Literal["rule", "llm", "fallback"]


class DecisionMetadata(BaseModel):
    """Structured attribution metadata for transparency in UI."""

    decision_source: DecisionSource = Field(
        default="fallback",
        description="Final decision path: rule (LLM skipped), llm, or fallback (timeouts/errors).",
    )
    contributors: list[str] = Field(
        default_factory=list,
        description="Ordered list of contributing components (e.g. cnn, rules).",
    )
    llm_used: bool = Field(default=False, description="Whether an LLM call was attempted.")
    cnn_used: bool = Field(default=False, description="Whether the CNN stack contributed a signal.")
    rules_used: bool = Field(default=True, description="Whether rule/policy logic contributed.")
    llm_status: Literal["used", "failed", "skipped"] = Field(default="skipped")
    llm_failure_reason: Optional[str] = Field(
        default=None,
        description="When llm_status=failed, a short reason (e.g. timeout).",
    )
    explanation: Optional[str] = Field(
        default=None,
        description="One-line explanation suitable for UI attribution display.",
    )


class ClaimRequest(BaseModel):
    """Inbound micro-insurance claim.

    Required fields are the three used for routing and policy checks; optional fields are
    forwarded to the fraud agent as additional context (``extra`` is forbidden on unknown keys).
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "claim_id": "MIC-2026-00412",
                "claim_amount": 75.5,
                "policy_limit": 500.0,
                "currency": "USD",
                "product_code": "PHONE_CRACK",
                "incident_date": "2026-03-28",
                "policyholder_id": "ph_8k2m",
                "description": "Dropped phone; screen cracked; repair quote from authorized shop.",
            }
        },
    )

    claim_id: str = Field(..., min_length=1, max_length=200, description="Stable id for this micro-claim.")
    claim_amount: float = Field(
        ...,
        ge=0,
        description="Requested payout amount (same currency unit as policy_limit).",
    )
    policy_limit: float = Field(
        ...,
        ge=0,
        description="Maximum benefit / cap for this micro-policy at triage time.",
    )
    currency: Optional[str] = Field(
        default=None,
        max_length=10,
        description="ISO 4217 code (e.g. USD); same unit as amounts.",
    )
    product_code: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Micro-product identifier (e.g. crop, phone, health bundle).",
    )
    incident_date: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Incident date (ISO 8601 date recommended).",
    )
    policyholder_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Internal id for the insured party.",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Free-text narrative for fraud / context (optional).",
    )
    rag_filter_decision: Optional[ClaimDecision] = Field(
        default=None,
        description="When RAG is enabled, restrict retrieval to stored claims with this decision (metadata).",
    )
    rag_metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional scalar equality metadata filter forwarded to the vector store (Chroma `where`).",
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Optional base64-encoded JPEG/PNG (data URLs allowed). Prefer multipart file `image` for large photos.",
    )

    @field_validator("image_base64")
    @classmethod
    def _validate_b64(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if s.startswith("data:") and "," in s:
            s = s.split(",", 1)[1].strip()
        try:
            raw = base64.b64decode(s, validate=True)
        except binascii.Error as exc:
            raise ValueError("image_base64 is not valid base64") from exc
        if len(raw) > 12 * 1024 * 1024:
            raise ValueError("image_base64 decodes to more than 12MB")
        return v


class ClaimProcessResponse(BaseModel):
    """Outcome of automated micro-insurance triage (approve, reject, or escalate)."""

    claim_id: str
    decision: ClaimDecision
    confidence_score: float = Field(ge=0.0, le=1.0)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    hitl_needed: bool = Field(default=False)
    review_reason: Optional[str] = Field(
        default=None,
        description="Why human review is (or is not) required for this decision.",
    )
    hitl_decision_reason: Optional[str] = Field(
        default=None,
        description="Audit-friendly reason string used for HITL routing.",
    )
    llm_used: bool = Field(default=False, description="Whether an LLM call was used for this claim.")
    fallback_used: bool = Field(
        default=False,
        description="Whether the pipeline fell back to deterministic logic (rule or fallback path).",
    )
    decision_source: DecisionSource = Field(
        default="fallback",
        description="Where the decision came from: rule (LLM skipped), llm, or fallback (timeouts/errors).",
    )
    metadata: Optional[DecisionMetadata] = Field(
        default=None,
        description="Structured attribution metadata (decision source + contributors).",
    )
    fraud_signal: Optional[str] = Field(
        default=None,
        description="Optional fraud signal flag (e.g. image_text_mismatch) for UI + logging.",
    )
    cnn_used: bool = Field(default=False, description="Whether the CNN output was used (cnn_label != 'unknown').")
    cnn_label: str = Field(default="unknown", description="CNN label (e.g. minor_crack).")
    cnn_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="CNN confidence in [0,1].")
    cnn_severity: str = Field(default="unknown", description="CNN severity label (low/medium/high/unknown).")
    agent_outputs: Dict[str, Any]


ReviewAction = Literal["APPROVED", "REJECTED"]


class ClaimReviewRequest(BaseModel):
    action: ReviewAction
    reviewed_by: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional reviewer id or label for the learning loop.",
    )


CaseStatus = Literal["NEW", "ASSIGNED", "IN_PROGRESS", "RESOLVED"]


class CaseAssignRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"assigned_to": "investigator_1"}})

    assigned_to: str = Field(..., min_length=1, max_length=200)


class CaseStatusUpdateRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"case_status": "IN_PROGRESS"}})

    case_status: CaseStatus


class CaseListItem(BaseModel):
    claim_id: str
    case_status: str
    assigned_to: str = ""
    decision: Optional[ClaimDecision] = None
    fraud_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: str = ""
    review_status: str = ""
    timestamp: str = ""


class CaseListResponse(BaseModel):
    cases: list[CaseListItem]


class ClaimListItem(BaseModel):
    claim_id: str
    claim_description: str = Field(default="")
    fraud_score: float = Field(default=0.0, ge=0.0, le=1.0)
    decision: Optional[ClaimDecision] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    explanation: Optional[str] = Field(default=None, description="Fraud analyst explanation (bullets or text).")
    review_status: Optional[ReviewAction] = Field(
        default=None,
        description="Human review outcome when present (feeds learning loop).",
    )
    hitl_needed: bool = Field(default=False)
    reviewed_action: Optional[ReviewAction] = None
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    entities: Optional[Dict[str, Any]] = Field(default=None, description="Structured hints from fraud agent.")
    has_image: bool = Field(default=False, description="Whether a photo was submitted with this claim.")
    image_damage_type: str = Field(default="", description="Image model damage label when available.")
    image_severity: str = Field(
        default="",
        description="Image model severity label when available (e.g. low / medium / high).",
    )
    llm_used: bool = Field(default=False, description="Whether an LLM call was used for this claim.")
    fallback_used: bool = Field(
        default=False,
        description="Whether the pipeline fell back to deterministic logic (rule or fallback path).",
    )
    decision_source: Optional[DecisionSource] = Field(
        default=None,
        description="rule (LLM skipped), llm, or fallback (timeouts/errors) when available.",
    )
    metadata: Optional[DecisionMetadata] = Field(
        default=None,
        description="Structured attribution metadata when available.",
    )
    fraud_signal: Optional[str] = Field(default=None, description="Optional fraud signal flag for the claim.")
    cnn_used: bool = Field(default=False, description="Whether the CNN output was used (cnn_label != 'unknown').")
    cnn_label: str = Field(default="", description="CNN label (e.g. minor_crack) when available.")
    cnn_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="CNN confidence in [0,1].")
    cnn_severity: str = Field(default="", description="CNN severity label when available (low/medium/high).")


class ClaimImagePreviewResponse(BaseModel):
    claim_id: str
    image_base64: str = Field(default="", description="JPEG preview bytes as standard base64 (no data URL prefix).")
    mime_type: str = Field(default="image/jpeg")
