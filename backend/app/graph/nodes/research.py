from typing import Any

from app.core.logging import get_logger
from app.graph.state import AppState

_logger = get_logger(__name__)

_NODE = "research"


async def research_node(state: AppState) -> dict[str, Any]:
    """Mark the research execution path and advance the step counter.

    The research path always proceeds to the retriever node so no
    retrieval logic lives here.  Future iterations may enrich the query
    with domain-specific context before passing control downstream.

    Writes
    ------
    current_node : str
    step_count   : incremented by 1
    """
    _logger.info("research_node_start", session_id=state["session_id"])
    return {
        "current_node": _NODE,
        "step_count": state.get("step_count", 0) + 1,
    }
