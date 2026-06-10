"""
LangGraph state definitions for the agentic workflow.

State design rules
------------------
* AppState is a TypedDict — the only type LangGraph can checkpoint and merge natively.
* Fields annotated with a reducer (operator.add) are *accumulated* across node writes;
  all other fields are *replaced* (last-write-wins within a single graph step).
* Nested structured types are also TypedDicts so they serialize to plain JSON, which is
  required by every checkpoint backend including PostgreSQL.
* Optional fields use NotRequired so the initial state dict does not need to supply them.
"""

import operator
from datetime import UTC, datetime
from typing import Annotated, Literal, NotRequired, TypedDict


# ── Primitive aliases ─────────────────────────────────────────────────────────

ApprovalStatus = Literal["pending", "approved", "rejected"]
RouteDecision = Literal["research", "support"]


# ── Document representation ───────────────────────────────────────────────────


class RetrievedDocument(TypedDict):
    """A single chunk returned from the vector store."""

    id: str
    content: str
    source: str
    metadata: dict[str, str]
    score: float  # cosine similarity from ChromaDB


class RankedDocument(TypedDict):
    """A retrieved document after CrossEncoder reranking."""

    id: str
    content: str
    source: str
    metadata: dict[str, str]
    retrieval_score: float  # original vector similarity
    rerank_score: float     # CrossEncoder relevance score


# ── Citation ──────────────────────────────────────────────────────────────────


class Citation(TypedDict):
    """Source reference produced by the generator node."""

    document_id: str
    source: str
    excerpt: str
    rerank_score: float


# ── Structured output ─────────────────────────────────────────────────────────


class StructuredOutput(TypedDict):
    """Strict schema for the response produced by the structured-output node.

    Mirrors the Pydantic schema in app.schemas.responses so the API layer
    can deserialise it without re-parsing.
    """

    summary: str
    answer: str
    citations: list[Citation]


# ── Groundedness evaluation ───────────────────────────────────────────────────


class EvaluatedClaim(TypedDict):
    """A single factual claim from the generated answer, with a support verdict."""

    claim: str
    supported: bool
    source_document_ids: list[str]
    reasoning: str


class GroundednessResult(TypedDict):
    """Aggregated groundedness evaluation produced by the groundedness node.

    groundedness_score  = len(supported_claims) / total_claims (0.0 when no claims).
    evaluated_at        ISO-8601 UTC timestamp of when the evaluation ran.
    """

    groundedness_score: float
    supported_claims: list[EvaluatedClaim]
    unsupported_claims: list[EvaluatedClaim]
    evaluated_at: str  # ISO-8601 UTC


# ── Confidence scores ─────────────────────────────────────────────────────────


class ConfidenceScores(TypedDict):
    """Confidence signals captured at each major stage of the workflow.

    router      LLM self-reported probability that the routing decision is correct.
    retrieval   Position-weighted mean of ChromaDB cosine-similarity scores.
    answer      Mean CrossEncoder rerank score of the context used for generation.
    overall     Weighted combination: router 20%, retrieval 30%, answer 50%.
    """

    router: float
    retrieval: float
    answer: float
    overall: float


# ── Final response ────────────────────────────────────────────────────────────


class FinalResponse(TypedDict):
    """Everything returned to the caller after approval."""

    session_id: str
    summary: str
    answer: str
    citations: list[Citation]
    route: RouteDecision
    approval_status: Literal["approved"]  # only "approved" responses reach here
    auto_approved: bool
    reviewer_id: str | None
    reviewer_comment: str | None
    created_at: str  # ISO-8601 UTC
    confidence: NotRequired[ConfidenceScores | None]
    groundedness: NotRequired[GroundednessResult | None]


# ── Error record ──────────────────────────────────────────────────────────────


class WorkflowError(TypedDict):
    """A single failure captured by a node.

    Using a list with operator.add as the reducer means every node can append
    errors without overwriting errors recorded by previous nodes.
    """

    node: str
    message: str
    timestamp: str  # ISO-8601 UTC — avoids datetime serialisation issues


# ── Approval metadata ─────────────────────────────────────────────────────────


class ApprovalRecord(TypedDict):
    """Written by the approval service once a reviewer acts."""

    reviewer_id: str
    action: ApprovalStatus
    comment: NotRequired[str]
    decided_at: str  # ISO-8601 UTC


# ── Primary state ─────────────────────────────────────────────────────────────


class AppState(TypedDict):
    """Single source of truth passed between every LangGraph node.

    Field contract
    --------------
    session_id, query
        Set once on entry; never mutated by nodes.

    route
        Written by the router node; read by the conditional edge that forks
        to research or support.

    retrieved_documents
        Written (replaced) by the retriever node.

    reranked_documents
        Written (replaced) by the reranker node.

    draft_response
        Raw LLM output from the generator node, including inline citation
        markers before structured parsing.

    structured_output
        Parsed and validated dict produced by the structured-output node.

    approval_status
        Starts as "pending".  Written by the human-approval node when a
        reviewer acts.  Drives the conditional edge after human_approval.

    approval_record
        Written by the approval service alongside approval_status.

    final_response
        Written by the final-response node; the value surfaced to the API.

    errors
        Annotated with operator.add so each node *appends* its errors rather
        than overwriting earlier ones.  A non-empty list does not halt the
        graph — nodes decide themselves whether to route to END.

    current_node
        Updated at the start of each node for observability / debugging.

    step_count
        Incremented by each node; guards against run-away loops.

    metadata
        Pass-through bag from the original API request.
    """

    # ── Required on entry ──────────────────────────────────────────────────────
    session_id: str
    query: str

    # ── Set by routing node ────────────────────────────────────────────────────
    route: NotRequired[RouteDecision | None]

    # ── Set by retrieval pipeline ──────────────────────────────────────────────
    retrieved_documents: NotRequired[list[RetrievedDocument]]
    reranked_documents: NotRequired[list[RankedDocument]]

    # ── Set by generation pipeline ─────────────────────────────────────────────
    draft_response: NotRequired[str | None]
    structured_output: NotRequired[StructuredOutput | None]

    # ── Set by approval flow ───────────────────────────────────────────────────
    approval_status: NotRequired[ApprovalStatus]
    approval_record: NotRequired[ApprovalRecord | None]
    auto_approved: NotRequired[bool]

    # ── Set by final-response node ─────────────────────────────────────────────
    final_response: NotRequired[FinalResponse | None]

    # ── Accumulated across all nodes (reducer = list concatenation) ────────────
    errors: Annotated[list[WorkflowError], operator.add]

    # ── Confidence scores (written by individual nodes) ────────────────────────
    router_confidence: NotRequired[float | None]      # from RouterAgent
    retrieval_confidence: NotRequired[float | None]   # aggregate similarity score
    answer_confidence: NotRequired[float | None]      # mean rerank score of context

    # ── Groundedness evaluation (written by groundedness node) ─────────────────
    groundedness: NotRequired[GroundednessResult | None]

    # ── Workflow tracking ──────────────────────────────────────────────────────
    current_node: NotRequired[str | None]
    step_count: NotRequired[int]

    # ── Pass-through from API request ─────────────────────────────────────────
    metadata: NotRequired[dict[str, str]]


# ── Convenience helpers ───────────────────────────────────────────────────────


def make_error(node: str, message: str) -> WorkflowError:
    """Construct a WorkflowError with the current UTC timestamp."""
    return WorkflowError(
        node=node,
        message=message,
        timestamp=datetime.now(UTC).isoformat(),
    )


def initial_state(session_id: str, query: str, metadata: dict[str, str] | None = None) -> AppState:
    """Return a valid AppState for the start of a new workflow run."""
    return AppState(
        session_id=session_id,
        query=query,
        route=None,
        retrieved_documents=[],
        reranked_documents=[],
        draft_response=None,
        structured_output=None,
        approval_status="pending",
        approval_record=None,
        auto_approved=False,
        final_response=None,
        errors=[],
        current_node=None,
        step_count=0,
        metadata=metadata or {},
        router_confidence=None,
        retrieval_confidence=None,
        answer_confidence=None,
        groundedness=None,
    )
