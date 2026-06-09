"""
Embedding service — wraps OllamaEmbeddings (nomic-embed-text).

The embeddings object is injectable so tests can replace it with a mock
without patching any imports.  All public methods are async; the underlying
Ollama HTTP call is awaited via aembed_* which langchain_ollama provides.
"""

from langchain_ollama import OllamaEmbeddings

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

_logger = get_logger(__name__)


class EmbeddingService:
    """Async wrapper around OllamaEmbeddings (nomic-embed-text).

    Parameters
    ----------
    embeddings:
        A LangChain Embeddings implementation.  Defaults to OllamaEmbeddings
        pointed at the configured Ollama instance.  Pass a mock in tests.
    """

    def __init__(self, embeddings: OllamaEmbeddings | None = None) -> None:
        settings = get_settings()
        self._embeddings: OllamaEmbeddings = embeddings or OllamaEmbeddings(
            model=settings.ollama.embedding_model,
            base_url=settings.ollama.base_url,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text.

        Raises
        ------
        EmbeddingError
            If the Ollama call fails for any reason.
        ValueError
            If *texts* is empty.
        """
        if not texts:
            raise ValueError("texts must not be empty")
        try:
            vectors = await self._embeddings.aembed_documents(texts)
        except Exception as exc:
            raise EmbeddingError(f"embed_documents failed: {exc}") from exc

        _logger.debug("embedded_documents", count=len(texts))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Return a single embedding vector for *text*.

        Raises
        ------
        EmbeddingError
            If the Ollama call fails for any reason.
        """
        if not text.strip():
            raise ValueError("text must not be blank")
        try:
            vector = await self._embeddings.aembed_query(text)
        except Exception as exc:
            raise EmbeddingError(f"embed_query failed: {exc}") from exc

        _logger.debug("embedded_query", dim=len(vector))
        return vector
