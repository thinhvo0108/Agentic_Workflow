"""
Support Agent — FAQ, troubleshooting, and customer support with adaptive retrieval.

Two-pass design
---------------
Pass 1  (triage)
    A fast LLM call produces a ConfidenceAssessment answering three questions:
    • Can the query be answered from general product/support knowledge?
    • How confident is the model (0–1)?
    • What category is the query? (faq | troubleshooting | general | requires_context)

Pass 2a  (direct answer — high confidence)
    If confidence ≥ CONFIDENCE_THRESHOLD AND answer_type ≠ "requires_context",
    a second LLM call generates the answer without consulting any documents.
    Citations will be empty.

Pass 2b  (retrieval-augmented — low confidence)
    Otherwise the agent uses the reranked documents already in state.
    Falls back to pass 2a when documents is empty so the workflow never deadlocks.

Output schema
-------------
SupportOutput subclasses ResearchOutput, adding retrieval_used and confidence.
Pydantic v2 ignores extra fields on model_validate, so structured_output_node
parses SupportOutput JSON via ResearchOutput.model_validate_json() unchanged.

Dependency injection
--------------------
Both internal chains (confidence + generation) are derived from the injected llm.
Pass a mock in tests to avoid touching Ollama.
"""

from typing import Literal

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.agents.research_agent import (
    ResearchOutput,
    _build_context,
    _override_citation_scores,
)
from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.graph.state import RankedDocument

_logger = get_logger(__name__)

# Queries with confidence below this threshold trigger retrieval-augmented generation.
CONFIDENCE_THRESHOLD: float = 0.7

# ── Prompts ────────────────────────────────────────────────────────────────────

_TRIAGE_SYSTEM = """\
You are a support triage classifier.  Evaluate whether the incoming query can be
answered accurately from general product knowledge alone, or requires consulting
the knowledge base.

Categories
----------
faq              : Common "how do I…" questions with well-known answers.
troubleshooting  : A known error or failure with a standard resolution path.
general          : A general product or feature question answerable from public docs.
requires_context : Unusual error codes, version-specific behaviour, account-specific
                   data, or any question that needs internal documentation to answer
                   reliably.

Rules
-----
• Set can_answer_directly = true only when you are confident (≥ 0.7) the answer
  is correct WITHOUT consulting any external documents.
• Set answer_type = "requires_context" when in doubt — it is always safe to look up.
• confidence must reflect how sure you are that the answer is accurate and complete.
"""

_DIRECT_SYSTEM = """\
You are a product support specialist.  Answer the support query concisely and
accurately using your general product knowledge and troubleshooting expertise.

Guidelines
----------
1. Be direct and action-oriented.
2. Use numbered steps for multi-step procedures.
3. If you are not completely certain of a detail, acknowledge the uncertainty.
4. Return an empty citations list — no external documents were consulted.
"""

_CONTEXT_SYSTEM = """\
You are a product support specialist.  Answer the support query using ONLY the
knowledge base articles provided below.

Guidelines
----------
1. Cite every factual claim with the document ID in square brackets, e.g. [doc-id].
2. Focus on concrete, actionable steps and verified solutions.
3. If the documents do not contain the information needed, say so explicitly and
   return an empty citations list.
4. Copy the document_id and source exactly as they appear in the document header.
"""

_HUMAN_TEMPLATE = """\
QUERY
-----
{query}

{context_section}"""


# ── Output schemas ─────────────────────────────────────────────────────────────


class ConfidenceAssessment(BaseModel):
    """Triage decision returned by the first LLM pass."""

    can_answer_directly: bool = Field(
        description="True if the query can be answered from general knowledge"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in answering correctly without documents",
    )
    answer_type: Literal["faq", "troubleshooting", "general", "requires_context"] = Field(
        description="Category of the support request"
    )
    reasoning: str = Field(
        min_length=5,
        description="One-sentence explanation of the triage decision",
    )


class SupportOutput(ResearchOutput):
    """ResearchOutput extended with support-specific observability fields.

    Because SupportOutput subclasses ResearchOutput:
    • isinstance(support_output, ResearchOutput) → True
    • ResearchOutput.model_validate_json(support_output.model_dump_json()) succeeds
      (Pydantic v2 ignores extra fields by default)
    • structured_output_node requires zero changes
    """

    retrieval_used: bool = Field(
        default=False,
        description="Whether retrieval-augmented generation was invoked",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Triage confidence score from the first LLM pass",
    )


# ── Pure helpers ───────────────────────────────────────────────────────────────


def _needs_retrieval(assessment: ConfidenceAssessment) -> bool:
    """Return True when the triage assessment indicates RAG should be used."""
    return (
        not assessment.can_answer_directly
        or assessment.confidence < CONFIDENCE_THRESHOLD
        or assessment.answer_type == "requires_context"
    )


# ── Agent ──────────────────────────────────────────────────────────────────────


class SupportAgent:
    """Generates structured support answers with adaptive retrieval.

    Parameters
    ----------
    llm:
        A LangChain BaseChatModel.  Defaults to ChatOllama from settings.
        Pass a mock in tests — both internal chains are derived from this instance.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        settings = get_settings()
        _llm: BaseChatModel = llm or ChatOllama(
            model=settings.ollama.default_model,
            base_url=settings.ollama.base_url,
            timeout=settings.ollama.timeout,
            temperature=0.0,
        )
        self._confidence_chain: Runnable = _llm.with_structured_output(ConfidenceAssessment)
        self._generate_chain: Runnable = _llm.with_structured_output(SupportOutput)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def generate(
        self,
        query: str,
        documents: list[RankedDocument],
    ) -> SupportOutput:
        """Generate a structured support answer with adaptive retrieval.

        Parameters
        ----------
        query:
            The user's support question.
        documents:
            Reranked knowledge-base chunks (may be empty).

        Returns
        -------
        SupportOutput
            Pydantic-validated output with retrieval_used and confidence populated.

        Raises
        ------
        LLMError
            If either LLM pass fails after all retries or returns an unexpected type.
        ValueError
            If *query* is blank.
        """
        if not query.strip():
            raise ValueError("query must not be blank")

        assessment = await self._assess_confidence(query)
        use_retrieval = _needs_retrieval(assessment) and bool(documents)

        if use_retrieval:
            result = await self._generate_with_context(query, documents)
            result = _override_citation_scores(result, documents)  # type: ignore[arg-type]
        else:
            result = await self._generate_direct(query)

        return result.model_copy(  # type: ignore[return-value]
            update={
                "retrieval_used": use_retrieval,
                "confidence": assessment.confidence,
            }
        )

    # ── Internal passes ────────────────────────────────────────────────────────

    async def _assess_confidence(self, query: str) -> ConfidenceAssessment:
        """Pass 1: triage the query and assess direct-answer confidence."""
        messages = [
            SystemMessage(content=_TRIAGE_SYSTEM),
            HumanMessage(content=query),
        ]
        try:
            result = await self._invoke_with_retry(self._confidence_chain, messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Confidence assessment failed: {exc}") from exc

        if not isinstance(result, ConfidenceAssessment):
            raise LLMError(
                f"Expected ConfidenceAssessment, got {type(result).__name__!r}"
            )
        _logger.info(
            "support_triage_complete",
            can_answer_directly=result.can_answer_directly,
            confidence=result.confidence,
            answer_type=result.answer_type,
        )
        return result

    async def _generate_direct(self, query: str) -> SupportOutput:
        """Pass 2a: answer from general knowledge, no documents."""
        messages = [
            SystemMessage(content=_DIRECT_SYSTEM),
            HumanMessage(
                content=_HUMAN_TEMPLATE.format(query=query, context_section="")
            ),
        ]
        try:
            result = await self._invoke_with_retry(self._generate_chain, messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Direct support generation failed: {exc}") from exc

        if not isinstance(result, SupportOutput):
            raise LLMError(
                f"Expected SupportOutput, got {type(result).__name__!r}"
            )
        return result

    async def _generate_with_context(
        self, query: str, documents: list[RankedDocument]
    ) -> SupportOutput:
        """Pass 2b: answer grounded in retrieved knowledge base articles."""
        context = _build_context(documents)
        messages = [
            SystemMessage(content=_CONTEXT_SYSTEM),
            HumanMessage(
                content=_HUMAN_TEMPLATE.format(
                    query=query,
                    context_section=f"DOCUMENTS\n---------\n{context}",
                )
            ),
        ]
        try:
            result = await self._invoke_with_retry(self._generate_chain, messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Context support generation failed: {exc}") from exc

        if not isinstance(result, SupportOutput):
            raise LLMError(
                f"Expected SupportOutput, got {type(result).__name__!r}"
            )
        return result

    # ── Retry helper ───────────────────────────────────────────────────────────

    async def _invoke_with_retry(self, chain: Runnable, messages: list) -> object:
        """Invoke *chain* with up to 3 attempts (exponential back-off 1–8 s)."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                return await chain.ainvoke(messages)
        raise LLMError("All retry attempts exhausted")  # pragma: no cover
