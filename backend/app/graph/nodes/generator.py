from app.graph.state import AppState


async def generator_node(state: AppState) -> dict:
    """Call the LLM to produce a cited draft answer from the reranked context.

    Reads
    -----
    query : str
    reranked_documents : list[RankedDocument]

    Writes
    ------
    draft_response : str   (raw LLM text with inline citation markers)
    current_node : str
    errors : list  (appended on failure)
    """
    raise NotImplementedError
