"""LLM-as-a-judge service.

Evaluates generated answers on four quality dimensions using a structured
LLM call. Mirrors the GroundednessEvaluator pattern:
  - LLM handles qualitative scoring per dimension (subjective classification)
  - We compute overall_score deterministically from dimension weights

Dimension weights
-----------------
faithfulness  0.40  — hallucinations are the highest risk
relevance     0.30  — must answer the actual question
completeness  0.20  — key aspects should be covered
coherence     0.10  — clarity and structure
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
from app.evaluation.judge_schemas import JudgeEvaluation
from app.graph.state import RankedDocument

_logger = get_logger(__name__)

_WEIGHTS: dict[str, float] = {
    "faithfulness": 0.40,
    "relevance": 0.30,
    "completeness": 0.20,
    "coherence": 0.10,
}

_SYSTEM_PROMPT = """\
You are an impartial judge evaluating an AI-generated answer against the source \
documents that were retrieved to produce it.

Score the answer on four dimensions from 0.0 (poor) to 1.0 (excellent):

FAITHFULNESS — Every factual claim in the answer is directly supported by the source documents.
  1.0  All claims are clearly backed by sources.
  0.5  Most claims are supported; one or two are reasonable inferences.
  0.0  Claims are unsupported or contradict the sources.

RELEVANCE — The answer directly and completely addresses the user's query.
  1.0  Fully addresses every aspect of the query.
  0.5  Addresses the main question but misses secondary aspects.
  0.0  Off-topic or only tangentially related.

COMPLETENESS — The answer covers all important aspects of the question.
  1.0  Comprehensive; no important aspect is missing.
  0.5  Covers the core answer but omits useful detail.
  0.0  Superficial; key information is absent.

COHERENCE — The answer is well-structured, clear, and internally consistent.
  1.0  Clear, logically ordered, no contradictions.
  0.5  Mostly clear but slightly disorganised or repetitive.
  0.0  Confusing, contradictory, or hard to follow.

Provide a one-sentence reasoning for each score and a brief critique (2-3 sentences) \
summarising the overall quality. Be strict and objective. Do not reward padding.
"""

_HUMAN_TEMPLATE = """\
USER QUERY
----------
{query}

SOURCE DOCUMENTS
----------------
{context}

GENERATED ANSWER
----------------
{answer}
"""


def compute_overall_score(evaluation: JudgeEvaluation) -> float:
    """Compute weighted overall score from dimension scores.

    Deterministic — never delegated to the LLM to avoid numeric inconsistency.
    """
    raw = (
        _WEIGHTS["faithfulness"] * evaluation.faithfulness.score
        + _WEIGHTS["relevance"] * evaluation.relevance.score
        + _WEIGHTS["completeness"] * evaluation.completeness.score
        + _WEIGHTS["coherence"] * evaluation.coherence.score
    )
    return round(max(0.0, min(1.0, raw)), 4)


def _build_context(documents: list[RankedDocument]) -> str:
    if not documents:
        return "(no source documents available)"
    parts: list[str] = []
    for doc in documents:
        parts.append(
            f"Document ID: {doc['id']}\nSource: {doc['source']}\n\n{doc['content']}\n{'─' * 60}"
        )
    return "\n\n".join(parts)


class LLMJudge:
    """Evaluates answer quality using a separate LLM call.

    Parameters
    ----------
    llm:
        A LangChain BaseChatModel. Defaults to ChatOllama from settings.
        Inject a mock in tests.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        settings = get_settings()
        _llm: BaseChatModel = llm or ChatOllama(  # type: ignore[call-arg]
            model=settings.ollama.default_model,
            base_url=settings.ollama.base_url,
            timeout=settings.ollama.timeout,
            temperature=0.0,
        )
        self._chain: Runnable[Any, Any] = _llm.with_structured_output(JudgeEvaluation)

    async def evaluate(
        self,
        query: str,
        answer: str,
        documents: list[RankedDocument],
    ) -> tuple[JudgeEvaluation, float]:
        """Run the LLM-as-a-judge evaluation.

        Returns
        -------
        tuple[JudgeEvaluation, float]
            Parsed evaluation object and the computed overall_score.

        Raises
        ------
        LLMError
            When the LLM call fails after all retries.
        """
        context = _build_context(documents)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=_HUMAN_TEMPLATE.format(query=query, answer=answer, context=context)
            ),
        ]

        try:
            result = await self._invoke_with_retry(messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LLM judge evaluation failed: {exc}") from exc

        if not isinstance(result, JudgeEvaluation):
            raise LLMError(f"Expected JudgeEvaluation, got {type(result).__name__!r}")

        overall = compute_overall_score(result)
        _logger.info(
            "llm_judge_evaluated",
            faithfulness=result.faithfulness.score,
            relevance=result.relevance.score,
            completeness=result.completeness.score,
            coherence=result.coherence.score,
            overall_score=overall,
        )
        return result, overall

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _invoke_with_retry(self, messages: list[Any]) -> object:
        return await self._chain.ainvoke(messages)
