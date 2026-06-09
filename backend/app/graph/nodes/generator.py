from app.agents.research_agent import ResearchAgent
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.graph.state import AppState, make_error

_logger = get_logger(__name__)

_NODE = "generator"


async def generator_node(state: AppState) -> dict:
    """Generate a cited draft answer using the ResearchAgent.

    The validated Pydantic output is serialised to JSON and stored as
    draft_response so the structured_output node can re-validate it as a
    TypedDict and downstream nodes have a stable JSON audit trail.

    Reads
    -----
    query               : str
    reranked_documents  : list[RankedDocument]

    Writes
    ------
    draft_response : str  (JSON-serialised ResearchOutput)
    current_node   : str
    step_count     : incremented by 1
    errors         : appended on failure
    """
    _logger.info(
        "generator_node_start",
        session_id=state["session_id"],
        doc_count=len(state.get("reranked_documents") or []),
    )

    step = state.get("step_count", 0) + 1

    try:
        agent = ResearchAgent()
        result = await agent.generate(
            query=state["query"],
            documents=state.get("reranked_documents") or [],
        )
    except LLMError as exc:
        _logger.error(
            "generator_node_failed",
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
            "generator_node_unexpected",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, f"Unexpected error: {exc}")],
        }

    draft = result.model_dump_json()
    _logger.info(
        "generator_node_done",
        session_id=state["session_id"],
        draft_len=len(draft),
    )
    return {
        "draft_response": draft,
        "current_node": _NODE,
        "step_count": step,
    }
