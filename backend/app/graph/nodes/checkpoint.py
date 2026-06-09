from app.graph.state import WorkflowState


async def checkpoint_node(state: WorkflowState) -> dict:
    """Persist full workflow state to PostgreSQL checkpoint store."""
    raise NotImplementedError
