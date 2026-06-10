"""
Knowledge-update node.

Runs after final_response, but only on the manual-approval path
(auto_approved=False).  Ingests the approved Q&A pair into ChromaDB so that
future queries on the same topic retrieve this document, boosting rerank
scores and increasing the likelihood of auto-approval next time.

Errors are caught and logged; the node always returns so the workflow
reaches END even if the KB write fails.
"""

from app.core.logging import get_logger
from app.graph.state import AppState, make_error

_logger = get_logger(__name__)
_NODE = "knowledge_update"


async def knowledge_update_node(state: AppState) -> dict:
    """Embed the approved Q&A pair and upsert it into the vector store."""
    from app.rag.ingestion import IngestDocument, IngestionPipeline

    step = state.get("step_count", 0) + 1
    final = state.get("final_response")
    query = (state.get("query") or "").strip()

    if not final or not query:
        _logger.warning("knowledge_update_skip_missing_data", session_id=state["session_id"])
        return {"knowledge_updated": False, "current_node": _NODE, "step_count": step}

    answer = (final.get("answer") or "").strip()
    if not answer:
        _logger.warning("knowledge_update_skip_empty_answer", session_id=state["session_id"])
        return {"knowledge_updated": False, "current_node": _NODE, "step_count": step}

    session_id = state["session_id"]
    reviewer_id = final.get("reviewer_id") or "human"

    # Store as plain knowledge-base prose.  Avoid "Question:/Answer:" labels —
    # those cause the LLM to transcribe the content verbatim rather than cite it,
    # leaving the citations array empty.  The query line still gives the vector
    # store enough signal to match semantically similar future queries.
    content = f"{query}\n\n{answer}"

    try:
        pipeline = IngestionPipeline()
        chunk_count = await pipeline.ingest([
            IngestDocument(
                content=content,
                source=f"approved_qa:{session_id}",
                metadata={
                    "type": "approved_qa",
                    "session_id": session_id,
                    "reviewer_id": reviewer_id,
                    "query": query[:256],
                },
            )
        ])
        _logger.info(
            "knowledge_update_done",
            session_id=session_id,
            chunk_count=chunk_count,
            reviewer_id=reviewer_id,
        )
        return {"knowledge_updated": True, "current_node": _NODE, "step_count": step}

    except Exception as exc:
        _logger.error(
            "knowledge_update_failed",
            session_id=session_id,
            error=str(exc),
        )
        return {
            "knowledge_updated": False,
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, str(exc))],
        }
