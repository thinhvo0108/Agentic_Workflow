"""
Retrieval service — queries ChromaDB and maps results to RetrievedDocument.

Score convention
----------------
ChromaDB stores vectors in cosine space (hnsw:space = "cosine").
The distance it returns is  1 − cosine_similarity  (range [0, 2]).
We convert to similarity with  score = max(0.0, 1.0 − distance)
so callers always receive a value in [0, 1] where 1 = perfect match.
"""

import asyncio
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError, RetrievalError
from app.core.logging import get_logger
from app.graph.state import RetrievedDocument
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStoreClient

_logger = get_logger(__name__)


def _distance_to_score(distance: float) -> float:
    """Convert ChromaDB cosine distance to a similarity score in [0, 1]."""
    return max(0.0, min(1.0, 1.0 - distance))


def _parse_chroma_results(results: dict[str, Any]) -> list[RetrievedDocument]:
    """Unpack the nested-list format ChromaDB returns into RetrievedDocument dicts."""
    ids: list[str] = results.get("ids", [[]])[0]
    documents: list[str | None] = results.get("documents", [[]])[0]
    metadatas: list[dict[str, Any] | None] = results.get("metadatas", [[]])[0]
    distances: list[float] = results.get("distances", [[]])[0]

    parsed: list[RetrievedDocument] = []
    for doc_id, content, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
        if content is None:
            continue
        meta: dict[str, str] = {k: str(v) for k, v in (metadata or {}).items()}
        parsed.append(
            RetrievedDocument(
                id=doc_id,
                content=content,
                source=meta.get("source", ""),
                metadata=meta,
                score=_distance_to_score(distance),
            )
        )
    return parsed


class RetrieverService:
    """Retrieves the top-K most semantically similar documents from ChromaDB.

    Parameters
    ----------
    vector_store:
        A VectorStoreClient (or compatible mock).
    embedding_service:
        An EmbeddingService (or compatible mock).
    """

    def __init__(
        self,
        vector_store: VectorStoreClient | None = None,
        embedding_service: EmbeddingService | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._settings = get_settings()
        self._store = vector_store or VectorStoreClient(collection_name=collection_name)
        self._embeddings = embedding_service or EmbeddingService()

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        """Return up to *top_k* documents most relevant to *query*.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Maximum number of results.  Defaults to settings.rag.retrieval_top_k.

        Raises
        ------
        EmbeddingError  – if the query embedding call fails.
        RetrievalError  – if the ChromaDB query fails.
        """
        k = top_k if top_k is not None else self._settings.rag.retrieval_top_k

        try:
            query_vector = await self._embeddings.embed_query(query)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Unexpected error embedding query: {exc}") from exc

        try:
            collection = await self._store.get_collection()
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"ChromaDB query failed: {exc}") from exc

        docs = _parse_chroma_results(results)
        _logger.info(
            "retrieval_complete",
            query_len=len(query),
            requested=k,
            returned=len(docs),
        )
        return docs
