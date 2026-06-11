"""Context precision evaluator — RAGAS Context Precision pillar.

Measures whether the retrieved documents are actually relevant to the query.

Approach
--------
The LLM receives the query and all reranked documents, then labels each
document relevant / irrelevant with one-sentence reasoning.

    context_precision_score = relevant_docs / total_docs  (0.0 if no docs)

This mirrors the GroundednessEvaluator pattern:
  - LLM does the qualitative binary classification per document
  - We compute the numeric score deterministically from the verdicts
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.evaluation.context_precision_schemas import ContextPrecisionEvaluation
from app.graph.state import RankedDocument

_logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a retrieval quality evaluator. Your task is to judge whether each retrieved \
document is relevant to answering the user's query.

A document is RELEVANT (is_relevant=true) if it contains information that would be \
useful or necessary to answer the query — even if only partially.

A document is IRRELEVANT (is_relevant=false) if it does not contain information \
related to the query, or if it is entirely off-topic.

For each document:
1. Read the document content carefully.
2. Decide: is_relevant = true or false.
3. Write exactly one sentence explaining your verdict.

Then write a brief overall_reasoning (1-2 sentences) summarising the retrieval quality.

You MUST produce one verdict per document, using the exact document_id provided.
"""

_HUMAN_TEMPLATE = """\
QUERY
-----
{query}

RETRIEVED DOCUMENTS
-------------------
{context}
"""


class ContextPrecisionEvaluator:
    """Evaluates whether retrieved documents are relevant to the query.

    Parameters
    ----------
    llm:
        A LangChain BaseChatModel. Defaults to ChatOllama from settings.
        Pass a mock instance in tests.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        settings = get_settings()
        _llm: BaseChatModel = llm or ChatOllama(  # type: ignore[call-arg]
            model=settings.ollama.default_model,
            base_url=settings.ollama.base_url,
            timeout=settings.ollama.timeout,
            temperature=0.0,
        )
        self._chain: Runnable[Any, Any] = _llm.with_structured_output(ContextPrecisionEvaluation)

    async def evaluate(
        self,
        query: str,
        documents: list[RankedDocument],
    ) -> ContextPrecisionEvaluation:
        """Run context precision evaluation.

        Parameters
        ----------
        query:
            The original user question.
        documents:
            Reranked source documents to evaluate.

        Returns
        -------
        ContextPrecisionEvaluation
            Structured output with per-document verdicts.

        Raises
        ------
        LLMError
            Propagated after all retries are exhausted.
        """
        context = _build_context(documents)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_HUMAN_TEMPLATE.format(query=query, context=context)),
        ]

        try:
            result = await self._invoke_with_retry(messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Context precision evaluation failed: {exc}") from exc

        if not isinstance(result, ContextPrecisionEvaluation):
            raise LLMError(f"Expected ContextPrecisionEvaluation, got {type(result).__name__!r}")

        relevant = sum(1 for v in result.verdicts if v.is_relevant)
        _logger.info(
            "context_precision_evaluated",
            total_docs=len(result.verdicts),
            relevant=relevant,
        )
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _invoke_with_retry(self, messages: list[Any]) -> object:
        return await self._chain.ainvoke(messages)


def _build_context(documents: list[RankedDocument]) -> str:
    if not documents:
        return "(no documents retrieved)"
    parts: list[str] = []
    for doc in documents:
        parts.append(
            f"Document ID: {doc['id']}\nSource: {doc['source']}\n\n{doc['content']}\n{'─' * 60}"
        )
    return "\n\n".join(parts)
