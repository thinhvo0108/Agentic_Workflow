from app.graph.state import AppState


async def support_node(state: AppState) -> dict:
    """Prepare context for the support path.

    Triggers retrieval when confidence is low.

    Writes
    ------
    current_node : str
    errors : list  (appended on failure)
    """
    raise NotImplementedError
