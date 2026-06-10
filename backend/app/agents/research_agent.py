"""
Research Agent — generates summary, answer, and citations from ranked documents.

Design
------
* ChatOllama is called via with_structured_output(ResearchOutput) so the LLM
  must return JSON that Pydantic validates before it enters the workflow state.
  Malformed output is surfaced as LLMError rather than silently corrupting state.

* rerank_score in CitationOutput is seeded by the agent from the actual document
  scores after the LLM call so the value is always authoritative — the model only
  needs to provide document_id, source, and excerpt.

* The llm argument enables full dependency injection: tests pass a mock without
  touching Ollama or HuggingFace.
"""

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.graph.state import RankedDocument

_logger = get_logger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a research assistant.  Your task is to answer the user's query using
ONLY the documents provided below.  Do not use external knowledge.

Rules
-----
1. Write a 1-2 sentence SUMMARY that captures the key finding.
2. Write a detailed ANSWER that directly addresses the query.
   Cite every factual claim with the document ID in square brackets, e.g. [doc-id].
3. For CITATIONS, include only documents you actually quoted or paraphrased.
   Copy the document_id and source exactly as shown.
   The excerpt must be a verbatim or near-verbatim quote (≥ 10 characters) that
   directly supports your answer.
4. If the documents do not contain enough information to answer the query,
   say so explicitly in the answer field and provide an empty citations list.
"""

_HUMAN_TEMPLATE = """\
QUERY
-----
{query}

DOCUMENTS
---------
{context}
"""


# ── Output schema ──────────────────────────────────────────────────────────────


class CitationOutput(BaseModel):
    """A single document citation produced by the LLM."""

    document_id: str = Field(description="Exact ID of the cited document")
    source: str = Field(description="Source identifier copied from the document header")
    excerpt: str = Field(
        min_length=5,
        description="Verbatim or near-verbatim quote from the document",
    )
    rerank_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="CrossEncoder relevance score — filled in by the agent, not the LLM",
    )


class ResearchOutput(BaseModel):
    """Structured response produced by the ResearchAgent."""

    summary: str = Field(
        min_length=10,
        description="1-2 sentence summary of the key finding",
    )
    answer: str = Field(
        min_length=20,
        description="Detailed answer with inline [document-id] citations",
    )
    citations: list[CitationOutput] = Field(
        default_factory=list,
        description="Documents cited in the answer",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_context(documents: list[RankedDocument]) -> str:
    """Format ranked documents into a numbered context block for the prompt."""
    if not documents:
        return "(no documents available)"

    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        parts.append(
            f"Document [{i}]\n"
            f"ID: {doc['id']}\n"
            f"Source: {doc['source']}\n"
            f"Relevance: {doc['rerank_score']:.4f}\n"
            f"\n"
            f"{doc['content']}\n"
            f"{'─' * 60}"
        )
    return "\n\n".join(parts)


def _override_citation_scores(
    output: ResearchOutput,
    documents: list[RankedDocument],
) -> ResearchOutput:
    """Replace LLM-generated rerank_scores with authoritative values from docs."""
    score_map = {doc["id"]: doc["rerank_score"] for doc in documents}
    patched = [
        c.model_copy(update={"rerank_score": score_map.get(c.document_id, c.rerank_score)})
        for c in output.citations
    ]
    return output.model_copy(update={"citations": patched})


# ── Agent ──────────────────────────────────────────────────────────────────────


class ResearchAgent:
    """Generates a structured research answer from a query and ranked documents.

    Parameters
    ----------
    llm:
        A LangChain BaseChatModel.  Defaults to ChatOllama from settings.
        Pass a mock instance in tests.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        settings = get_settings()
        _llm: BaseChatModel = llm or ChatOllama(
            model=settings.ollama.default_model,
            base_url=settings.ollama.base_url,
            timeout=settings.ollama.timeout,
            temperature=0.0,
            think=False,
        )
        self._chain: Runnable = _llm.with_structured_output(ResearchOutput)

    async def generate(
        self,
        query: str,
        documents: list[RankedDocument],
    ) -> ResearchOutput:
        """Generate a structured answer grounded in *documents*.

        Parameters
        ----------
        query:
            The user's original research question.
        documents:
            Reranked documents ordered by relevance (most relevant first).

        Returns
        -------
        ResearchOutput
            Pydantic-validated structured response with summary, answer, citations.

        Raises
        ------
        LLMError
            If the chain raises after all retries or returns an unexpected type.
        """
        context = _build_context(documents)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=_HUMAN_TEMPLATE.format(query=query, context=context)
            ),
        ]

        try:
            result = await self._invoke_with_retry(messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Research generation failed: {exc}") from exc

        if not isinstance(result, ResearchOutput):
            raise LLMError(
                f"Expected ResearchOutput from LLM, got {type(result).__name__!r}"
            )

        result = _override_citation_scores(result, documents)

        _logger.info(
            "research_agent_generated",
            summary_len=len(result.summary),
            answer_len=len(result.answer),
            citation_count=len(result.citations),
        )
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _invoke_with_retry(self, messages: list) -> object:
        return await self._chain.ainvoke(messages)
