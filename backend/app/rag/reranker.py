from langchain_core.documents import Document

from app.core.config import get_settings


class RerankerService:
    """Uses BAAI/bge-reranker-large CrossEncoder to rerank retrieved documents."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model: object | None = None

    def _load_model(self) -> None:
        raise NotImplementedError

    async def rerank(self, query: str, documents: list[Document], top_n: int | None = None) -> list[Document]:
        raise NotImplementedError
