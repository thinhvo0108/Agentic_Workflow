from app.graph.state import AppState, FinalResponse


async def final_response_node(state: AppState) -> dict:
    """Assemble and store the approved FinalResponse.

    Reads
    -----
    session_id, structured_output, route, approval_status

    Writes
    ------
    final_response : FinalResponse
    current_node : str
    errors : list  (appended on failure)
    """
    raise NotImplementedError
