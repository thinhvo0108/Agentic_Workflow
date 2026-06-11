from typing import Any

from app.agents.research_agent import ResearchOutput
from app.core.logging import get_logger
from app.graph.state import AppState, Citation, StructuredOutput, make_error
from app.services.confidence import score_answer

_logger = get_logger(__name__)

_NODE = "structured_output"


async def structured_output_node(state: AppState) -> dict[str, Any]:
    """Validate the draft JSON and convert it to a StructuredOutput TypedDict.

    The draft produced by the generator node is already Pydantic-validated JSON,
    but we re-validate here so the node acts as an explicit schema gate: any
    downstream code that reads structured_output can trust its shape.

    Reads
    -----
    draft_response : str  (JSON-serialised ResearchOutput)

    Writes
    ------
    structured_output : StructuredOutput
    current_node      : str
    step_count        : incremented by 1
    errors            : appended on failure
    """
    _logger.info("structured_output_node_start", session_id=state["session_id"])

    step = state.get("step_count", 0) + 1
    draft = state.get("draft_response")

    if not draft:
        _logger.error("structured_output_no_draft", session_id=state["session_id"])
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, "No draft_response found in state")],
        }

    try:
        parsed = ResearchOutput.model_validate_json(draft)
    except Exception as exc:
        _logger.error(
            "structured_output_validation_failed",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, f"Schema validation failed: {exc}")],
        }

    output = StructuredOutput(
        summary=parsed.summary,
        answer=parsed.answer,
        citations=[
            Citation(
                document_id=c.document_id,
                source=c.source,
                excerpt=c.excerpt,
                rerank_score=c.rerank_score,
            )
            for c in parsed.citations
        ],
    )

    answer_conf = score_answer(state.get("reranked_documents") or [])
    _logger.info(
        "structured_output_node_done",
        session_id=state["session_id"],
        citation_count=len(output["citations"]),
        answer_confidence=answer_conf,
    )
    return {
        "structured_output": output,
        "answer_confidence": answer_conf,
        "current_node": _NODE,
        "step_count": step,
    }
