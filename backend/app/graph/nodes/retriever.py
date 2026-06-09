from app.graph.state import AppState, RetrievedDocument


async def retriever_node(state: AppState) -> dict:
    """Query ChromaDB for the top-K most relevant document chunks.

    Reads
    -----
    query : str

    Writes
    ------
    retrieved_documents : list[RetrievedDocument]
    current_node : str
    errors : list  (appended on failure)
    """
    raise NotImplementedError
