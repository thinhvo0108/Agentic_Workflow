from app.graph.state import WorkflowState


async def generator_node(state: WorkflowState) -> dict:
    """Generate a cited draft answer from the reranked documents."""
    raise NotImplementedError
