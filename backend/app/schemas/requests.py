import re
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

_UNSAFE_PATTERN = re.compile(r"[<>{};`$]")

QueryStr = Annotated[str, Field(min_length=1, max_length=4096)]


class WorkflowRequest(BaseModel):
    query: QueryStr
    session_id: str | None = Field(default=None, description="Resume an existing workflow session")
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        if _UNSAFE_PATTERN.search(v):
            raise ValueError("Query contains disallowed characters")
        return v.strip()

    @field_validator("metadata")
    @classmethod
    def sanitize_metadata(cls, v: dict[str, str]) -> dict[str, str]:
        return {k[:64]: val[:256] for k, val in v.items()}


class ApprovalRequest(BaseModel):
    session_id: str
    action: str = Field(pattern="^(approved|rejected)$")
    reviewer_id: str = Field(min_length=1, max_length=128)
    comment: str | None = Field(default=None, max_length=1024)
    edited_answer: str | None = Field(default=None, max_length=16_384)


class IngestDocumentRequest(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    source: str = Field(min_length=1, max_length=256)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def sanitize_metadata(cls, v: dict[str, str]) -> dict[str, str]:
        return {k[:64]: val[:256] for k, val in v.items()}


class IngestRequest(BaseModel):
    documents: list[IngestDocumentRequest] = Field(min_length=1, max_length=100)
    agent_type: str | None = Field(
        default=None,
        description=(
            "Target agent collection: 'research', 'support', or any agent route string. "
            "Documents are stored in 'knowledge_base_{agent_type}' so they are only "
            "retrieved by that agent. Omit to write to the shared default collection."
        ),
        max_length=64,
    )
