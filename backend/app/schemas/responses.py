from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_id: str
    source: str
    excerpt: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class ConfidenceScores(BaseModel):
    """Per-stage confidence signals for an completed workflow run."""

    router: float = Field(ge=0.0, le=1.0, description="Routing classification confidence (LLM self-reported)")
    retrieval: float = Field(ge=0.0, le=1.0, description="Position-weighted mean of similarity scores")
    answer: float = Field(ge=0.0, le=1.0, description="Mean rerank score of context used for generation")
    overall: float = Field(ge=0.0, le=1.0, description="Weighted combination: router 20%, retrieval 30%, answer 50%")


class WorkflowResponse(BaseModel):
    session_id: str
    summary: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: Literal["research", "support"]
    approval_status: Literal["pending", "approved", "rejected"]
    confidence: ConfidenceScores | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkflowStatusResponse(BaseModel):
    session_id: str
    status: Literal["running", "awaiting_approval", "approved", "rejected", "failed", "completed"]
    current_node: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ApprovalResponse(BaseModel):
    session_id: str
    action: Literal["approved", "rejected"]
    reviewer_id: str
    comment: str | None = None
    processed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
