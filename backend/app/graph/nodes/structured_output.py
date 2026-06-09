from app.graph.state import WorkflowState


async def structured_output_node(state: WorkflowState) -> dict:
    """Parse the draft into a strict Pydantic-validated structured response."""
    raise NotImplementedError
