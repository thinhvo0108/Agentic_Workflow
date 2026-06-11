from typing import Any

from app.agents.router import RouterAgent
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.graph.state import AppState, make_error
from app.observability.token_tracker import TokenCounterCallback, instrumented_llm

_logger = get_logger(__name__)

_NODE = "router"


async def router_node(state: AppState) -> dict[str, Any]:
    """Classify the user query and write the routing decision into state.

    Returns a partial state update — LangGraph merges it with the existing
    state, so only the keys listed below change.

    Writes
    ------
    route         : "research" | "support"
    current_node  : "router"
    step_count    : incremented by 1
    errors        : appended on failure (reducer concatenates)
    """
    _logger.info("router_node_start", session_id=state["session_id"])

    step = state.get("step_count", 0) + 1

    try:
        counter = TokenCounterCallback()
        agent = RouterAgent(llm=instrumented_llm(counter))
        result = await agent.classify(state["query"])
    except LLMError as exc:
        _logger.error(
            "router_node_failed",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, str(exc))],
        }
    except Exception as exc:
        _logger.error(
            "router_node_unexpected_error",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, f"Unexpected error: {exc}")],
        }

    _logger.info(
        "router_node_done",
        session_id=state["session_id"],
        route=result.route,
        confidence=result.confidence,
    )
    return {
        "route": result.route,
        "router_confidence": result.confidence,
        "current_node": _NODE,
        "step_count": step,
        "total_tokens": counter.total,
    }
