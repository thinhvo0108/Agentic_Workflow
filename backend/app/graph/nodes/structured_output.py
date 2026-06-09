from app.graph.state import AppState, StructuredOutput


async def structured_output_node(state: AppState) -> dict:
    """Parse the draft into the strict StructuredOutput schema.

    Reads
    -----
    draft_response : str
    reranked_documents : list[RankedDocument]

    Writes
    ------
    structured_output : StructuredOutput
    current_node : str
    errors : list  (appended on failure)
    """
    raise NotImplementedError
