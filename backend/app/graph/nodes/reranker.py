from app.graph.state import WorkflowState


async def reranker_node(state: WorkflowState) -> dict:
    """Rerank retrieved documents using CrossEncoder and return top-N."""
    raise NotImplementedError
