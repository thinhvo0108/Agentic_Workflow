from langchain_core.documents import Document

from app.core.config import get_settings


class RetrieverService:
    """Queries ChromaDB for the top-K most relevant document chunks."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        raise NotImplementedError

    async def add_documents(self, documents: list[Document]) -> None:
        raise NotImplementedError
