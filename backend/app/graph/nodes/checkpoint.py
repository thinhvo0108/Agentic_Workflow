from app.graph.state import AppState


async def checkpoint_node(state: AppState) -> dict:
    """Persist the current state snapshot to PostgreSQL.

    LangGraph's built-in checkpointer stores the full state graph
    automatically; this node handles any *application-level* persistence
    (e.g. writing a human-readable row to an audit table).

    Reads
    -----
    session_id, query, route, retrieved_documents, reranked_documents,
    draft_response, structured_output

    Writes
    ------
    current_node : str
    errors : list  (appended on failure)
    """
    raise NotImplementedError
