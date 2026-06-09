"""
Domain models for the application-level checkpoint repository.

These are separate from LangGraph's own checkpoint schema, which is managed
by langgraph-checkpoint-postgres.  The models here drive the workflow_checkpoints
audit table that stores a human-readable summary of each major pipeline stage.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class CheckpointStage(StrEnum):
    """Pipeline stages at which a checkpoint record is written."""

    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    GENERATION = "generation"
    APPROVAL = "approval"
    FINAL = "final"


class CheckpointRecord(BaseModel):
    """A single row in the workflow_checkpoints audit table.

    id and created_at are populated by PostgreSQL; they are None until a
    record is saved and the RETURNING clause fills them in.
    """

    id: int | None = None
    session_id: str
    stage: CheckpointStage
    query: str
    route: str | None = None
    retrieved_doc_count: int = Field(default=0, ge=0)
    reranked_doc_count: int = Field(default=0, ge=0)
    has_draft: bool = False
    has_structured_output: bool = False
    approval_status: str | None = None
    error_count: int = Field(default=0, ge=0)
    # Curated JSON snapshot — not the raw state (which can be megabytes).
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
