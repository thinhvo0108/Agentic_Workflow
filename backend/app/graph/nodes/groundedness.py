"""Groundedness evaluation node.

Runs after structured_output and before checkpoint so the score is
available in the checkpoint and surfaced in the final API response.

The node is non-blocking on failure: if the LLM call fails, the error is
appended to state["errors"] and the workflow continues to checkpoint.
Downstream nodes can inspect state["groundedness"] for None to detect a
failed evaluation.
"""

from app.core.logging import get_logger
from app.evaluation.service import evaluate_groundedness
from app.graph.state import AppState, make_error

_logger = get_logger(__name__)

_NODE = "groundedness"


async def groundedness_node(state: AppState) -> dict:
    """Evaluate how well the generated answer is grounded in source documents.

    Reads
    -----
    query, structured_output, reranked_documents

    Writes
    ------
    groundedness   : GroundednessResult (None on failure)
    current_node   : str
    step_count     : incremented by 1
    errors         : appended on failure
    """
    _logger.info("groundedness_node_start", session_id=state["session_id"])

    step = state.get("step_count", 0) + 1
    so = state.get("structured_output")

    if so is None:
        _logger.warning(
            "groundedness_node_skip_no_output", session_id=state["session_id"]
        )
        return {"current_node": _NODE, "step_count": step}

    answer = so.get("answer", "")
    if not answer:
        _logger.warning(
            "groundedness_node_skip_empty_answer", session_id=state["session_id"]
        )
        return {"current_node": _NODE, "step_count": step}

    documents = state.get("reranked_documents") or []

    try:
        result = await evaluate_groundedness(
            query=state["query"],
            answer=answer,
            documents=documents,
        )
    except Exception as exc:
        _logger.error(
            "groundedness_node_failed",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, f"Groundedness evaluation failed: {exc}")],
        }

    _logger.info(
        "groundedness_node_done",
        session_id=state["session_id"],
        score=result["groundedness_score"],
        supported=len(result["supported_claims"]),
        unsupported=len(result["unsupported_claims"]),
    )
    return {
        "groundedness": result,
        "current_node": _NODE,
        "step_count": step,
    }
