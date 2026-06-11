from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class JudgeDimensionScore(BaseModel):
    """Score and reasoning for one LLM-as-a-judge evaluation dimension."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class WorkflowMetrics(BaseModel):
    """Observability snapshot for a completed workflow run.

    All four headline metrics are always present (tokens may be 0 if Ollama
    did not return token counts). hallucination_rate and judge_score are None
    when the corresponding evaluation node did not run.
    """

    started_at: str = Field(description="ISO-8601 UTC — workflow submission time")
    completed_at: str = Field(description="ISO-8601 UTC — final-response time")
    latency_ms: float = Field(ge=0.0, description="Wall-clock latency in milliseconds")
    total_tokens: int = Field(
        ge=0, description="Sum of prompt + completion tokens across all LLM calls"
    )
    error_count: int = Field(ge=0, description="Number of nodes that recorded errors")
    error_rate: float = Field(ge=0.0, le=1.0, description="error_count / step_count")
    hallucination_rate: float | None = Field(
        default=None,
        description="Fraction of unsupported claims (1 - groundedness_score); None if groundedness did not run",
    )
    judge_score: float | None = Field(
        default=None,
        description="LLM-as-a-judge overall score; None if judge did not run",
    )
    context_precision_score: float | None = Field(
        default=None,
        description="Fraction of retrieved docs relevant to the query; None if context_eval did not run",
    )
    step_count: int = Field(ge=0, description="Total nodes executed")


class JudgeResult(BaseModel):
    """LLM-as-a-judge evaluation of the generated answer quality.

    overall_score is a deterministic weighted average:
        faithfulness 40%, relevance 30%, completeness 20%, coherence 10%

    recommendation is derived from overall_score:
        "auto_approve" (>= 0.70) | "needs_review" (< 0.70)
    """

    faithfulness: JudgeDimensionScore
    relevance: JudgeDimensionScore
    completeness: JudgeDimensionScore
    coherence: JudgeDimensionScore
    overall_score: float = Field(ge=0.0, le=1.0)
    recommendation: Literal["auto_approve", "needs_review"]
    critique: str = Field(description="2-3 sentence holistic evaluation from the judge")
    evaluated_at: str = Field(description="ISO-8601 UTC timestamp of the evaluation")


class Citation(BaseModel):
    document_id: str
    source: str
    excerpt: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class ConfidenceScores(BaseModel):
    """Per-stage confidence signals for a completed workflow run."""

    router: float = Field(
        ge=0.0, le=1.0, description="Routing classification confidence (LLM self-reported)"
    )
    retrieval: float = Field(
        ge=0.0, le=1.0, description="Position-weighted mean of similarity scores"
    )
    answer: float = Field(
        ge=0.0, le=1.0, description="Mean rerank score of context used for generation"
    )
    overall: float = Field(
        ge=0.0, le=1.0, description="Weighted combination: router 20%, retrieval 30%, answer 50%"
    )


class EvaluatedClaim(BaseModel):
    """A single factual claim from the answer, with a groundedness verdict."""

    claim: str
    supported: bool
    source_document_ids: list[str] = Field(default_factory=list)
    reasoning: str


class GroundednessResult(BaseModel):
    """LLM-based evaluation of whether the answer is grounded in source documents."""

    groundedness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of supported claims: len(supported) / total_claims",
    )
    supported_claims: list[EvaluatedClaim] = Field(default_factory=list)
    unsupported_claims: list[EvaluatedClaim] = Field(default_factory=list)
    evaluated_at: str = Field(description="ISO-8601 UTC timestamp of the evaluation")


class DocumentRelevanceVerdict(BaseModel):
    """Relevance verdict for a single retrieved document."""

    document_id: str
    is_relevant: bool
    reasoning: str


class ContextPrecisionResult(BaseModel):
    """Context precision evaluation — RAGAS Context Precision pillar."""

    context_precision_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of retrieved docs relevant to the query: relevant / total",
    )
    relevant_documents: list[DocumentRelevanceVerdict] = Field(default_factory=list)
    irrelevant_documents: list[DocumentRelevanceVerdict] = Field(default_factory=list)
    evaluated_at: str = Field(description="ISO-8601 UTC timestamp of the evaluation")


class WebSearchResult(BaseModel):
    """A single DuckDuckGo result fetched to assist the human reviewer."""

    title: str
    link: str
    snippet: str


class DraftResponse(BaseModel):
    """Structured output available for human review at the approval gate."""

    session_id: str
    query: str
    route: Literal["research", "support"]
    summary: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceScores | None = None
    groundedness: GroundednessResult | None = None
    context_precision: ContextPrecisionResult | None = None
    judge_result: JudgeResult | None = None
    web_search_results: list[WebSearchResult] = Field(default_factory=list)


class WorkflowResponse(BaseModel):
    session_id: str
    summary: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: Literal["research", "support"]
    approval_status: Literal["pending", "approved", "rejected"]
    auto_approved: bool = False
    knowledge_updated: bool = False
    reviewer_id: str | None = None
    reviewer_comment: str | None = None
    confidence: ConfidenceScores | None = None
    groundedness: GroundednessResult | None = None
    context_precision: ContextPrecisionResult | None = None
    judge_result: JudgeResult | None = None
    metrics: WorkflowMetrics | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkflowStatusResponse(BaseModel):
    session_id: str
    status: Literal[
        "running", "awaiting_approval", "approved", "rejected", "failed", "completed", "not_found"
    ]
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
