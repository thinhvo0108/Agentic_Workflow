"""
Ingestion pipeline — splits, embeds, and indexes documents into ChromaDB.

Pipeline
--------
IngestDocument list
  → text splitting (fixed-size chunks with overlap)
  → embedding in configurable batches
  → upsert into ChromaDB (deterministic IDs avoid duplicates on re-ingest)

ID scheme
---------
Each chunk receives a deterministic SHA-256-based ID derived from its
source, index within the document, and the first 128 characters of content.
Re-ingesting the same content therefore updates existing entries rather
than creating duplicates.
"""

import asyncio
import hashlib
from typing import TypedDict

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError, RetrievalError
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStoreClient

_logger = get_logger(__name__)

_EMBED_BATCH_SIZE = 64  # chunks embedded in one Ollama call


# ── Input schema ──────────────────────────────────────────────────────────────


class IngestDocument(TypedDict):
    """A single source document submitted to the ingestion pipeline."""

    content: str
    source: str
    metadata: dict[str, str]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split *text* into overlapping fixed-size character chunks.

    Breaks are made at the last whitespace boundary inside the window so
    chunks never cut a word in half.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        next_start = end - chunk_overlap
        if next_start <= start:
            # Guarantee forward progress on pathological input
            next_start = start + max(1, chunk_size - chunk_overlap)
        start = next_start

    return chunks


def _make_chunk_id(source: str, chunk_index: int, content: str) -> str:
    """Return a 32-character deterministic hex ID for a chunk."""
    raw = f"{source}::{chunk_index}::{content[:128]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _make_chunk_metadata(doc: IngestDocument, chunk_index: int) -> dict[str, str]:
    return {
        "source": doc["source"],
        "chunk_index": str(chunk_index),
        **{k: str(v) for k, v in doc.get("metadata", {}).items()},
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────


class IngestionPipeline:
    """Splits documents into chunks, embeds them, and upserts into ChromaDB.

    Parameters
    ----------
    vector_store:
        VectorStoreClient instance (or mock for testing).
    embedding_service:
        EmbeddingService instance (or mock for testing).
    chunk_size:
        Maximum characters per chunk.
    chunk_overlap:
        Character overlap between adjacent chunks.
    """

    def __init__(
        self,
        vector_store: VectorStoreClient | None = None,
        embedding_service: EmbeddingService | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        collection_name: str | None = None,
    ) -> None:
        self._settings = get_settings()
        self._store = vector_store or VectorStoreClient(collection_name=collection_name)
        self._embeddings = embedding_service or EmbeddingService()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ── Public API ─────────────────────────────────────────────────────────────

    async def ingest(self, documents: list[IngestDocument]) -> int:
        """Ingest *documents* and return the total number of chunks stored.

        Raises
        ------
        ValueError     – if documents is empty.
        EmbeddingError – if embedding a batch fails.
        RetrievalError – if writing to ChromaDB fails.
        """
        if not documents:
            raise ValueError("documents must not be empty")

        # 1. Split every document into chunks
        flat: list[tuple[IngestDocument, int, str]] = []
        for doc in documents:
            for idx, chunk in enumerate(
                _chunk_text(doc["content"], self._chunk_size, self._chunk_overlap)
            ):
                flat.append((doc, idx, chunk))

        if not flat:
            _logger.warning("ingest_no_chunks", doc_count=len(documents))
            return 0

        # 2. Embed + upsert in batches
        total_stored = 0
        for batch_start in range(0, len(flat), _EMBED_BATCH_SIZE):
            batch = flat[batch_start : batch_start + _EMBED_BATCH_SIZE]
            stored = await self._upsert_batch(batch)
            total_stored += stored

        _logger.info(
            "ingestion_complete",
            doc_count=len(documents),
            chunk_count=total_stored,
        )
        return total_stored

    async def delete_source(self, source: str) -> int:
        """Delete all chunks whose metadata source equals *source*.

        Returns the number of chunks deleted.
        """
        try:
            collection = await self._store.get_collection()
            existing = await asyncio.to_thread(
                collection.get,
                where={"source": source},
                include=[],
            )
            ids: list[str] = existing.get("ids", [])
            if ids:
                await asyncio.to_thread(collection.delete, ids=ids)
            _logger.info("source_deleted", source=source, chunk_count=len(ids))
            return len(ids)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"delete_source failed for '{source}': {exc}") from exc

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _upsert_batch(
        self, batch: list[tuple[IngestDocument, int, str]]
    ) -> int:
        texts = [chunk for _, _, chunk in batch]
        ids = [_make_chunk_id(doc["source"], idx, chunk) for doc, idx, chunk in batch]
        metadatas = [_make_chunk_metadata(doc, idx) for doc, idx, _ in batch]

        try:
            embeddings = await self._embeddings.embed_documents(texts)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Batch embedding failed: {exc}") from exc

        try:
            collection = await self._store.get_collection()
            await asyncio.to_thread(
                collection.upsert,
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"ChromaDB upsert failed: {exc}") from exc

        return len(batch)
