"""LLM token usage tracker.

Uses a LangChain callback to collect token counts from Ollama's response
metadata.  Ollama returns two fields in generation_info for each completion:

    prompt_eval_count   — tokens in the prompt (input)
    eval_count          — tokens in the completion (output)

Usage at node level
-------------------
Create one TokenCounterCallback per node invocation, pass it to ChatOllama
via the `callbacks` parameter, then read `.total` after the LLM call returns.
Because all agents accept an optional `llm` argument, no agent code changes:

    counter = TokenCounterCallback()
    llm = _instrumented_llm(counter)
    agent = RouterAgent(llm=llm)
    await agent.classify(query)
    tokens_this_node = counter.total

The AppState field `total_tokens` uses operator.add as its reducer, so each
node's return value is automatically summed into the running total.
"""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.core.config import get_settings
from app.core.logging import get_logger

_logger = get_logger(__name__)


class TokenCounterCallback(BaseCallbackHandler):
    """Accumulates prompt + completion tokens from Ollama LLM responses.

    Thread/task safety: one instance per node invocation — no shared state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token counts from Ollama's generation_info."""
        for gen_list in response.generations:
            for gen in gen_list:
                info: dict[str, Any] = getattr(gen, "generation_info", None) or {}
                self.prompt_tokens += info.get("prompt_eval_count", 0)
                self.completion_tokens += info.get("eval_count", 0)

        _logger.debug(
            "token_counter_update",
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total=self.total,
        )


def instrumented_llm(counter: TokenCounterCallback) -> Any:
    """Return a ChatOllama instance with the token counter wired in.

    Passes the counter as a callback so on_llm_end fires for every LLM call
    made by any chain built from this LLM instance (including with_structured_output).
    """
    from langchain_ollama import ChatOllama

    settings = get_settings()
    return ChatOllama(  # type: ignore[call-arg]
        model=settings.ollama.default_model,
        base_url=settings.ollama.base_url,
        timeout=settings.ollama.timeout,
        temperature=0.0,
        callbacks=[counter],
    )
