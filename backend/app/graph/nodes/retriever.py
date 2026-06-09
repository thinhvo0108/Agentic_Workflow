from app.core.exceptions import EmbeddingError, RetrievalError
from app.core.logging import get_logger
from app.graph.state import AppState, make_error
from app.rag.retriever import RetrieverService
from app.services.confidence import score_retrieval

_logger = get_logger(__name__)

_NODE = "retriever"


async def retriever_node(state: AppState) -> dict:
    """Query ChromaDB for the top-K chunks most relevant to the user query.

    Reads
    -----
    query : str

    Writes
    ------
    retrieved_documents : list[RetrievedDocument]
    current_node        : str
    step_count          : incremented by 1
    errors              : appended on failure
    """
    _logger.info("retriever_node_start", session_id=state["session_id"])

    step = state.get("step_count", 0) + 1

    try:
        service = RetrieverService()
        docs = await service.retrieve(state["query"])
    except (EmbeddingError, RetrievalError) as exc:
        _logger.error(
            "retriever_node_failed",
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
            "retriever_node_unexpected",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, f"Unexpected error: {exc}")],
        }

    retrieval_conf = score_retrieval(docs)
    _logger.info(
        "retriever_node_done",
        session_id=state["session_id"],
        doc_count=len(docs),
        retrieval_confidence=retrieval_conf,
    )
    return {
        "retrieved_documents": docs,
        "retrieval_confidence": retrieval_conf,
        "current_node": _NODE,
        "step_count": step,
    }
