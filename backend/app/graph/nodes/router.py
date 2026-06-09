from app.graph.state import WorkflowState


async def router_node(state: WorkflowState) -> dict:
    """Determine user intent and route to research or support agent."""
    raise NotImplementedError
