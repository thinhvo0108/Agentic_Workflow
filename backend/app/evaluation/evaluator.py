"""LLM-based groundedness evaluator.

Groundedness measures whether the generated answer is supported by the
retrieved source documents.  Rather than asking the LLM to output a raw
float (which is unreliable), we ask it to:

  1. Extract every factual claim from the answer.
  2. Label each claim supported / unsupported, citing which document IDs
     back it up.

The groundedness_score is then computed deterministically:
    score = len(supported_claims) / total_claims  (0.0 if no claims)

This approach produces consistent scores because the hard work (claim
classification) is done by the LLM on a binary scale, while the numeric
aggregation is ours.
"""

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.evaluation.schemas import GroundednessEvaluation
from app.graph.state import RankedDocument

_logger = get_logger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a groundedness evaluator.  Your task is to check whether a generated
answer is supported by the provided source documents.

Steps
-----
1. Read the ANSWER carefully.
2. Extract every distinct factual CLAIM from the answer.  A claim is any
   statement that could in principle be verified (not questions, opinions,
   or acknowledgements of uncertainty).
3. For each claim decide: is it SUPPORTED by at least one source document?
   - supported=true   → the document clearly states or implies the claim.
   - supported=false  → no document backs this claim (or it contradicts it).
4. For supported claims, list the document IDs that provide backing.
5. Write one-sentence reasoning for each verdict.
6. Write an overall_reasoning summarising the answer's groundedness.

Be strict: partial or vague support does not count as supported.
"""

_HUMAN_TEMPLATE = """\
QUERY
-----
{query}

ANSWER
------
{answer}

SOURCE DOCUMENTS
----------------
{context}
"""


# ── Evaluator ─────────────────────────────────────────────────────────────────


class GroundednessEvaluator:
    """Evaluates whether a generated answer is grounded in retrieved documents.

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
        self._chain: Runnable = _llm.with_structured_output(GroundednessEvaluation)

    async def evaluate(
        self,
        query: str,
        answer: str,
        documents: list[RankedDocument],
    ) -> GroundednessEvaluation:
        """Run the LLM-based groundedness check.

        Parameters
        ----------
        query:
            The original user question (context for the evaluator).
        answer:
            The generated answer text to evaluate.
        documents:
            The reranked source documents available to the answer generator.

        Returns
        -------
        GroundednessEvaluation
            Structured output with per-claim verdicts and overall reasoning.

        Raises
        ------
        LLMError
            Propagated from the LLM chain after all retries are exhausted.
        """
        context = _build_context(documents)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=_HUMAN_TEMPLATE.format(
                    query=query, answer=answer, context=context
                )
            ),
        ]

        try:
            result = await self._invoke_with_retry(messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Groundedness evaluation failed: {exc}") from exc

        if not isinstance(result, GroundednessEvaluation):
            raise LLMError(
                f"Expected GroundednessEvaluation from LLM, got {type(result).__name__!r}"
            )

        supported = sum(1 for c in result.claims if c.supported)
        _logger.info(
            "groundedness_evaluated",
            total_claims=len(result.claims),
            supported=supported,
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


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_context(documents: list[RankedDocument]) -> str:
    if not documents:
        return "(no source documents available)"
    parts: list[str] = []
    for doc in documents:
        parts.append(
            f"Document ID: {doc['id']}\n"
            f"Source: {doc['source']}\n"
            f"\n"
            f"{doc['content']}\n"
            f"{'─' * 60}"
        )
    return "\n\n".join(parts)
