from app.core.logging import get_logger
from app.graph.state import AppState

_logger = get_logger(__name__)

_NODE = "support"


async def support_node(state: AppState) -> dict:
    """Mark the support execution path and advance the step counter.

    The support path always proceeds through the retriever so documents are
    available when the generator node calls SupportAgent.generate().  The
    confidence check that decides whether those documents are *used* lives
    inside SupportAgent itself, not here.

    Writes
    ------
    current_node : str
    step_count   : incremented by 1
    """
    _logger.info("support_node_start", session_id=state["session_id"])
    return {
        "current_node": _NODE,
        "step_count": state.get("step_count", 0) + 1,
    }
