from app.graph.state import AppState, RouteDecision


async def router_node(state: AppState) -> dict:
    """Classify the user query and set the routing decision.

    Writes
    ------
    route : RouteDecision
        "research" or "support"
    current_node : str
    errors : list  (appended on failure)
    """
    raise NotImplementedError
