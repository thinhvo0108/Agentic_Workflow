from app.core.config import get_settings
from app.graph.state import RouteDecision


class RouterAgent:
    """LLM-based intent classifier that determines which agent should handle a query."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def classify(self, query: str) -> RouteDecision:
        raise NotImplementedError
