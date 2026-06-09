from app.graph.state import WorkflowState


async def retriever_node(state: WorkflowState) -> dict:
    """Retrieve top-K chunks from the vector database."""
    raise NotImplementedError
