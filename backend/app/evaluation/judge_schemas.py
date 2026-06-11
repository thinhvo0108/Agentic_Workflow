"""Pydantic schemas for LLM-as-a-judge structured output.

The LLM scores each dimension independently. Overall score is computed
deterministically by the service after parsing — consistent with the
groundedness evaluator pattern (LLM does classification; we do aggregation).
"""

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    """Score and one-sentence reasoning for a single evaluation dimension."""

    score: float = Field(ge=0.0, le=1.0, description="Quality from 0.0 (poor) to 1.0 (excellent)")
    reasoning: str = Field(description="One-sentence justification for the score")


class JudgeEvaluation(BaseModel):
    """Structured output produced by the LLM-as-a-judge.

    Dimension weights used for overall_score (computed post-LLM):
        faithfulness  0.40  — no hallucinations is the highest priority
        relevance     0.30  — must answer the actual question
        completeness  0.20  — coverage of key aspects
        coherence     0.10  — clarity and structure
    """

    faithfulness: DimensionScore = Field(
        description="Every factual claim is directly supported by source documents"
    )
    relevance: DimensionScore = Field(
        description="Answer directly and fully addresses the user query"
    )
    completeness: DimensionScore = Field(
        description="All important aspects of the question are covered"
    )
    coherence: DimensionScore = Field(
        description="Answer is well-structured, clear, and internally consistent"
    )
    critique: str = Field(description="2-3 sentence holistic evaluation of the answer quality")
