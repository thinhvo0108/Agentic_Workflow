from app.core.config import get_settings


class SupportAgent:
    """Agent for FAQ, customer support, and troubleshooting.

    Invokes retrieval when confidence falls below threshold.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def run(self, query: str) -> dict:
        raise NotImplementedError
