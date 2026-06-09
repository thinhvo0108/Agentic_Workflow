from app.core.config import get_settings


class ResearchAgent:
    """Agent for product research, technical questions, and knowledge exploration.

    Always invokes the retrieval pipeline before generating a response.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def run(self, query: str) -> dict:
        raise NotImplementedError
