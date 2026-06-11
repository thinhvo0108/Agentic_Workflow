from typing import Any

from app.core.exceptions import RerankingError
from app.core.logging import get_logger
from app.graph.state import AppState, make_error
from app.rag.reranker import RerankerService

_logger = get_logger(__name__)

_NODE = "reranker"


async def reranker_node(state: AppState) -> dict[str, Any]:
    """Score retrieved documents with the CrossEncoder and keep top-N.

    Reads
    -----
    query                : str
    retrieved_documents  : list[RetrievedDocument]

    Writes
    ------
    reranked_documents : list[RankedDocument]
    current_node       : str
    step_count         : incremented by 1
    errors             : appended on failure
    """
    _logger.info(
        "reranker_node_start",
        session_id=state["session_id"],
        doc_count=len(state.get("retrieved_documents") or []),
    )

    step = state.get("step_count", 0) + 1

    try:
        service = RerankerService()
        ranked = await service.rerank(
            query=state["query"],
            documents=state.get("retrieved_documents") or [],
        )
    except (RerankingError, ValueError) as exc:
        _logger.error(
            "reranker_node_failed",
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
            "reranker_node_unexpected",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, f"Unexpected error: {exc}")],
        }

    _logger.info(
        "reranker_node_done",
        session_id=state["session_id"],
        ranked_count=len(ranked),
    )
    return {
        "reranked_documents": ranked,
        "current_node": _NODE,
        "step_count": step,
    }
