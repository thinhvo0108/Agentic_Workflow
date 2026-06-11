"""
Router agent — classifies an incoming query into "research" or "support".

Design
------
* Uses ChatOllama via LangChain's with_structured_output(), which instructs
  the model to return valid JSON matching the RouteOutput schema.
* The LLM is injected via the constructor so tests can pass a mock without
  patching any imports.
* _invoke_with_retry wraps the raw chain call with tenacity so transient
  Ollama timeouts are retried before surfacing an LLMError.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.graph.state import RouteDecision

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a request classifier for an AI assistant platform.
Classify each incoming user query into exactly one of two routing categories.

RESEARCH — Use when the query requires:
  • In-depth knowledge retrieval or synthesis
  • Technical explanations, architecture, or design analysis
  • Comparative analysis or market research
  • Learning about how something works conceptually

  Examples:
  "Explain how RAFT consensus works."
  "Compare SQL vs NoSQL for time-series workloads."
  "What are best practices for microservices authentication?"
  "Research recent advances in transformer architectures."

SUPPORT — Use when the query is:
  • A how-to or operational question about a product or service
  • A troubleshooting request or error report
  • An account, billing, or service-status question
  • A simple factual lookup with a known, direct answer

  Examples:
  "How do I reset my password?"
  "I'm getting a 500 error when calling the login endpoint."
  "Where can I find my invoices?"
  "Is the API currently down?"

When ambiguous, prefer RESEARCH for analytical intent and SUPPORT for
operational / actionable intent.

Respond with your route, a confidence score between 0 and 1, and a short
one-sentence reasoning."""


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class RouteOutput(BaseModel):
    """Structured classification result returned by the router LLM."""

    route: RouteDecision = Field(description="Destination agent: 'research' or 'support'")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in the classification")
    reasoning: str = Field(min_length=1, description="One-sentence explanation of the decision")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class RouterAgent:
    """LLM-based intent classifier.

    Parameters
    ----------
    llm:
        A LangChain BaseChatModel instance.  When None (the default in
        production) the agent builds a ChatOllama from settings.  Passing an
        explicit instance allows tests to inject a mock without any patching.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        settings = get_settings()
        _llm: BaseChatModel = llm or ChatOllama(  # type: ignore[call-arg]
            model=settings.ollama.default_model,
            base_url=settings.ollama.base_url,
            timeout=settings.ollama.timeout,
        )
        # with_structured_output returns a Runnable that outputs a RouteOutput
        self._chain: Runnable[Any, Any] = _llm.with_structured_output(RouteOutput)

    async def classify(self, query: str) -> RouteOutput:
        """Classify *query* and return a RouteOutput.

        Raises
        ------
        LLMError
            If the chain raises after all retries, or if the model returns
            an unexpected type.
        """
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        try:
            result = await self._invoke_with_retry(messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Router classification failed: {exc}") from exc

        if not isinstance(result, RouteOutput):
            raise LLMError(f"Expected RouteOutput from LLM, got {type(result).__name__!r}")

        _logger.info(
            "router_classified",
            route=result.route,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
        return result

    # Retry up to 3 times on transient exceptions (network, timeout).
    # LLMError is our own final error — we never want to retry that.
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _invoke_with_retry(self, messages: list[Any]) -> object:
        return await self._chain.ainvoke(messages)
