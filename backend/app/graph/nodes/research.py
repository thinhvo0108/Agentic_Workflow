from app.graph.state import AppState


async def research_node(state: AppState) -> dict:
    """Prepare context for the research path.

    Always triggers retrieval.  May enrich the query with domain context
    before handing off to the retriever node.

    Writes
    ------
    current_node : str
    errors : list  (appended on failure)
    """
    raise NotImplementedError
