"""Pydantic schemas for LLM-structured context precision evaluation output."""

from pydantic import BaseModel, Field


class DocumentRelevanceVerdict(BaseModel):
    """Relevance verdict for a single retrieved document."""

    document_id: str = Field(
        description="The document ID being evaluated (match the ID given in the context)"
    )
    is_relevant: bool = Field(
        description="True if this document contains information useful for answering the query"
    )
    reasoning: str = Field(
        description="One sentence explaining why the document is or is not relevant to the query"
    )


class ContextPrecisionEvaluation(BaseModel):
    """Full structured output from the context precision LLM call."""

    verdicts: list[DocumentRelevanceVerdict] = Field(
        description="One relevance verdict per retrieved document, in the same order as provided"
    )
    overall_reasoning: str = Field(
        description="Brief overall assessment of retrieval quality for this query"
    )
