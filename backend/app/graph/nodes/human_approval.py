from app.graph.state import WorkflowState


async def human_approval_node(state: WorkflowState) -> dict:
    """Pause workflow and wait for a human reviewer to approve or reject."""
    raise NotImplementedError
