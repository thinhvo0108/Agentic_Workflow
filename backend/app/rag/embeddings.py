from app.core.config import get_settings


class EmbeddingService:
    """Wraps Ollama nomic-embed-text to produce document and query embeddings."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError
