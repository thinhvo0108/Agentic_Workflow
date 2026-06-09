from app.graph.state import AppState, RankedDocument


async def reranker_node(state: AppState) -> dict:
    """Score retrieved documents with the CrossEncoder and keep top-N.

    Reads
    -----
    query : str
    retrieved_documents : list[RetrievedDocument]

    Writes
    ------
    reranked_documents : list[RankedDocument]
    current_node : str
    errors : list  (appended on failure)
    """
    raise NotImplementedError
