from app.graph.state import WorkflowState


async def final_response_node(state: WorkflowState) -> dict:
    """Assemble and return the approved final response."""
    raise NotImplementedError
