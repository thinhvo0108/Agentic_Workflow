"""Pydantic schemas for LLM-structured groundedness evaluation output."""

from pydantic import BaseModel, Field


class ClaimVerdict(BaseModel):
    """Verdict for a single factual claim extracted from the generated answer."""

    claim: str = Field(description="The factual claim extracted verbatim from the answer")
    supported: bool = Field(
        description="True if at least one source document clearly supports this claim"
    )
    source_document_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source documents that support this claim (empty if unsupported)",
    )
    reasoning: str = Field(
        description="One sentence explaining why the claim is or is not supported"
    )


class GroundednessEvaluation(BaseModel):
    """Full structured output produced by the GroundednessEvaluator LLM call."""

    claims: list[ClaimVerdict] = Field(
        description="All factual claims found in the answer, each with a support verdict"
    )
    overall_reasoning: str = Field(
        description="Brief overall assessment of how well the answer is grounded in sources"
    )
