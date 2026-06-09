from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


ApprovalStatus = Literal["pending", "approved", "rejected"]
RouteDecision = Literal["research", "support"]
NodeStatus = Literal["pending", "running", "completed", "failed"]


class NodeResult(BaseModel):
    node: str
    status: NodeStatus
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WorkflowState(BaseModel):
    # ── Input ──────────────────────────────────────────────────────────────────
    session_id: str
    query: str
    metadata: dict[str, str] = Field(default_factory=dict)

    # ── Routing ────────────────────────────────────────────────────────────────
    route: RouteDecision | None = None

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieved_documents: list[Document] = Field(default_factory=list)
    reranked_documents: list[Document] = Field(default_factory=list)

    # ── Generation ─────────────────────────────────────────────────────────────
    draft: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)

    # ── Structured output ──────────────────────────────────────────────────────
    summary: str | None = None
    answer: str | None = None

    # ── Approval ───────────────────────────────────────────────────────────────
    approval_status: ApprovalStatus = "pending"
    reviewer_id: str | None = None
    reviewer_comment: str | None = None

    # ── Workflow tracking ──────────────────────────────────────────────────────
    current_node: str | None = None
    node_results: list[NodeResult] = Field(default_factory=list)
    error: str | None = None
    step_count: int = 0

    # ── Timestamps ─────────────────────────────────────────────────────────────
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
