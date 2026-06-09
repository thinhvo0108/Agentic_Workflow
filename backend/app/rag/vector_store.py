"""
ChromaDB client wrapper — manages the connection and collection lifecycle.

Design notes
------------
* All ChromaDB SDK calls are synchronous.  They are dispatched to a thread
  pool via asyncio.to_thread() so the async event loop is never blocked.
* The client and collection are lazily initialised on first use and then
  cached for the lifetime of the instance.
* An injectable `client` parameter lets tests supply an EphemeralClient
  without network access.
"""

import asyncio
from typing import Any

import chromadb
import chromadb.errors

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger

_logger = get_logger(__name__)

# Cosine distance is 1 − cos(θ).  Storing vectors in cosine space means the
# nearest-neighbour query returns documents sorted by semantic similarity.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class VectorStoreClient:
    """Thin async wrapper around a ChromaDB HTTP or ephemeral client.

    Parameters
    ----------
    client:
        A chromadb client instance (HttpClient, EphemeralClient, …).
        When None the client is built from settings at first use.
    """

    def __init__(self, client: Any | None = None) -> None:
        self._settings = get_settings()
        self._raw_client: Any | None = client
        self._collection: Any | None = None

    # ── Lazy initialisation ────────────────────────────────────────────────────

    def _get_raw_client(self) -> Any:
        if self._raw_client is None:
            self._raw_client = chromadb.HttpClient(
                host=self._settings.chroma.host,
                port=self._settings.chroma.port,
            )
        return self._raw_client

    async def get_collection(self) -> Any:
        """Return the configured collection, creating it if absent."""
        if self._collection is not None:
            return self._collection

        client = self._get_raw_client()
        try:
            self._collection = await asyncio.to_thread(
                client.get_or_create_collection,
                name=self._settings.chroma.collection_name,
                metadata=_COLLECTION_METADATA,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Failed to initialise ChromaDB collection "
                f"'{self._settings.chroma.collection_name}': {exc}"
            ) from exc

        _logger.info(
            "chroma_collection_ready",
            collection=self._settings.chroma.collection_name,
        )
        return self._collection

    # ── Health ─────────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Return True if ChromaDB is reachable."""
        try:
            client = self._get_raw_client()
            await asyncio.to_thread(client.heartbeat)
            return True
        except Exception:
            return False

    # ── Collection stats ───────────────────────────────────────────────────────

    async def count(self) -> int:
        """Return the number of documents in the collection."""
        collection = await self.get_collection()
        return await asyncio.to_thread(collection.count)
