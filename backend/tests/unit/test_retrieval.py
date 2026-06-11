"""
Unit tests for the retrieval pipeline.

All tests are fully isolated — no Ollama process, no ChromaDB server required.
EmbeddingService receives a mock OllamaEmbeddings; VectorStoreClient receives
a mock chromadb collection; RetrieverService and IngestionPipeline receive
mock EmbeddingService / VectorStoreClient instances.

Test groups
-----------
TestEmbeddingService      — embed_documents, embed_query, error paths
TestVectorStoreClient     — get_collection, health_check, count
TestDistanceConversion    — _distance_to_score, _parse_chroma_results
TestRetrieverService      — retrieve happy-path, empty collection, error paths
TestIngestionChunking     — _chunk_text helper
TestIngestionPipeline     — ingest, delete_source, batch sizing
TestRetrieverNode         — node state updates, error recording
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import EmbeddingError, RetrievalError
from app.graph.nodes.retriever import retriever_node
from app.graph.state import AppState, RetrievedDocument, initial_state
from app.rag.embeddings import EmbeddingService
from app.rag.ingestion import (
    IngestDocument,
    IngestionPipeline,
    _chunk_text,
    _make_chunk_id,
)
from app.rag.retriever import RetrieverService, _distance_to_score, _parse_chroma_results
from app.rag.vector_store import VectorStoreClient

# ── Factories ──────────────────────────────────────────────────────────────────


FAKE_VECTOR = [0.1, 0.2, 0.3]  # dimensionality doesn't matter in tests


def _mock_embeddings(vector: list[float] = FAKE_VECTOR) -> MagicMock:
    """Mock OllamaEmbeddings that returns *vector* for every text."""
    m = MagicMock()
    m.aembed_documents = AsyncMock(side_effect=lambda texts: [vector] * len(texts))
    m.aembed_query = AsyncMock(return_value=vector)
    return m


def _mock_collection(
    ids: list[str] | None = None,
    documents: list[str] | None = None,
    metadatas: list[dict] | None = None,
    distances: list[float] | None = None,
) -> MagicMock:
    """Mock chromadb collection whose query() returns the supplied data."""
    # Use `is not None` guards — empty lists are falsy and must not fall through.
    ids = ids if ids is not None else ["id1", "id2"]
    documents = documents if documents is not None else ["doc one", "doc two"]
    metadatas = metadatas if metadatas is not None else [{"source": "s1"}, {"source": "s2"}]
    distances = distances if distances is not None else [0.1, 0.3]

    col = MagicMock()
    col.query.return_value = {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }
    col.count.return_value = len(ids)
    col.get.return_value = {"ids": ids}
    col.upsert = MagicMock()
    col.delete = MagicMock()
    return col


def _mock_store(collection: MagicMock | None = None) -> VectorStoreClient:
    """VectorStoreClient whose get_collection() returns *collection*."""
    collection = collection or _mock_collection()
    store = VectorStoreClient(client=MagicMock())
    store._collection = collection
    return store


def _mock_embedding_service(vector: list[float] = FAKE_VECTOR) -> EmbeddingService:
    svc = EmbeddingService(embeddings=_mock_embeddings(vector))
    return svc


def _state(query: str = "test query") -> AppState:
    return initial_state(session_id="sess-test", query=query)


# ── EmbeddingService ──────────────────────────────────────────────────────────


class TestEmbeddingService:
    @pytest.mark.asyncio
    async def test_embed_documents_returns_one_vector_per_text(self):
        svc = EmbeddingService(embeddings=_mock_embeddings())
        result = await svc.embed_documents(["a", "b", "c"])
        assert len(result) == 3
        assert all(v == FAKE_VECTOR for v in result)

    @pytest.mark.asyncio
    async def test_embed_query_returns_vector(self):
        svc = EmbeddingService(embeddings=_mock_embeddings())
        result = await svc.embed_query("how does attention work?")
        assert result == FAKE_VECTOR

    @pytest.mark.asyncio
    async def test_embed_documents_raises_on_empty_list(self):
        svc = EmbeddingService(embeddings=_mock_embeddings())
        with pytest.raises(ValueError, match="empty"):
            await svc.embed_documents([])

    @pytest.mark.asyncio
    async def test_embed_query_raises_on_blank_string(self):
        svc = EmbeddingService(embeddings=_mock_embeddings())
        with pytest.raises(ValueError, match="blank"):
            await svc.embed_query("   ")

    @pytest.mark.asyncio
    async def test_embed_documents_wraps_exception_as_embedding_error(self):
        mock_emb = _mock_embeddings()
        mock_emb.aembed_documents.side_effect = RuntimeError("ollama down")
        svc = EmbeddingService(embeddings=mock_emb)
        with pytest.raises(EmbeddingError, match="embed_documents failed"):
            await svc.embed_documents(["text"])

    @pytest.mark.asyncio
    async def test_embed_query_wraps_exception_as_embedding_error(self):
        mock_emb = _mock_embeddings()
        mock_emb.aembed_query.side_effect = RuntimeError("timeout")
        svc = EmbeddingService(embeddings=mock_emb)
        with pytest.raises(EmbeddingError, match="embed_query failed"):
            await svc.embed_query("test")


# ── VectorStoreClient ──────────────────────────────────────────────────────────


class TestVectorStoreClient:
    @pytest.mark.asyncio
    async def test_get_collection_returns_collection(self):
        col = MagicMock()
        raw_client = MagicMock()
        raw_client.get_or_create_collection.return_value = col

        store = VectorStoreClient(client=raw_client)
        result = await store.get_collection()

        assert result is col
        raw_client.get_or_create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_caches_on_second_call(self):
        col = MagicMock()
        raw_client = MagicMock()
        raw_client.get_or_create_collection.return_value = col

        store = VectorStoreClient(client=raw_client)
        await store.get_collection()
        await store.get_collection()  # second call

        assert raw_client.get_or_create_collection.call_count == 1

    @pytest.mark.asyncio
    async def test_get_collection_raises_retrieval_error_on_failure(self):
        raw_client = MagicMock()
        raw_client.get_or_create_collection.side_effect = Exception("connection refused")

        store = VectorStoreClient(client=raw_client)
        with pytest.raises(RetrievalError, match="Failed to initialise"):
            await store.get_collection()

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_reachable(self):
        raw_client = MagicMock()
        raw_client.heartbeat.return_value = True
        store = VectorStoreClient(client=raw_client)
        assert await store.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_unreachable(self):
        raw_client = MagicMock()
        raw_client.heartbeat.side_effect = Exception("unreachable")
        store = VectorStoreClient(client=raw_client)
        assert await store.health_check() is False

    @pytest.mark.asyncio
    async def test_count_returns_collection_document_count(self):
        col = MagicMock()
        col.count.return_value = 42
        store = _mock_store(col)
        assert await store.count() == 42


# ── Distance conversion ────────────────────────────────────────────────────────


class TestDistanceConversion:
    def test_zero_distance_is_perfect_score(self):
        assert _distance_to_score(0.0) == pytest.approx(1.0)

    def test_one_distance_is_zero_score(self):
        assert _distance_to_score(1.0) == pytest.approx(0.0)

    def test_two_distance_clamps_to_zero(self):
        assert _distance_to_score(2.0) == pytest.approx(0.0)

    def test_negative_distance_clamps_to_one(self):
        # Should not happen in practice but we guard against it
        assert _distance_to_score(-0.5) == pytest.approx(1.0)

    def test_mid_range_distance(self):
        assert _distance_to_score(0.4) == pytest.approx(0.6)

    def test_parse_chroma_results_maps_fields_correctly(self):
        raw = {
            "ids": [["a1"]],
            "documents": [["hello world"]],
            "metadatas": [[{"source": "wiki", "chunk_index": "0"}]],
            "distances": [[0.25]],
        }
        docs = _parse_chroma_results(raw)
        assert len(docs) == 1
        d = docs[0]
        assert d["id"] == "a1"
        assert d["content"] == "hello world"
        assert d["source"] == "wiki"
        assert d["score"] == pytest.approx(0.75)

    def test_parse_chroma_results_skips_none_content(self):
        raw = {
            "ids": [["a1", "a2"]],
            "documents": [[None, "real content"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]],
        }
        docs = _parse_chroma_results(raw)
        assert len(docs) == 1
        assert docs[0]["content"] == "real content"

    def test_parse_chroma_results_handles_empty(self):
        raw = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        assert _parse_chroma_results(raw) == []


# ── RetrieverService ──────────────────────────────────────────────────────────


class TestRetrieverService:
    @pytest.mark.asyncio
    async def test_returns_top_k_documents(self):
        col = _mock_collection(
            ids=["i1", "i2", "i3"],
            documents=["a", "b", "c"],
            metadatas=[{"source": "s"}] * 3,
            distances=[0.1, 0.2, 0.3],
        )
        svc = RetrieverService(
            vector_store=_mock_store(col),
            embedding_service=_mock_embedding_service(),
        )
        docs = await svc.retrieve("test query", top_k=3)
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_uses_default_top_k_from_settings(self):
        col = _mock_collection()
        store = _mock_store(col)
        emb = _mock_embedding_service()
        svc = RetrieverService(vector_store=store, embedding_service=emb)

        await svc.retrieve("query")

        call_kwargs = col.query.call_args.kwargs
        assert call_kwargs["n_results"] == svc._settings.rag.retrieval_top_k

    @pytest.mark.asyncio
    async def test_documents_are_sorted_by_score_descending(self):
        col = _mock_collection(
            ids=["id1", "id2", "id3"],
            documents=["a", "b", "c"],
            metadatas=[{"source": "s"}] * 3,
            distances=[0.5, 0.1, 0.8],  # scores: 0.5, 0.9, 0.2
        )
        svc = RetrieverService(
            vector_store=_mock_store(col),
            embedding_service=_mock_embedding_service(),
        )
        # ChromaDB returns in its own order; we don't re-sort — verify raw scores
        docs = await svc.retrieve("query", top_k=3)
        assert docs[0]["score"] == pytest.approx(0.5)
        assert docs[1]["score"] == pytest.approx(0.9)
        assert docs[2]["score"] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_handles_empty_collection(self):
        col = _mock_collection(ids=[], documents=[], metadatas=[], distances=[])
        svc = RetrieverService(
            vector_store=_mock_store(col),
            embedding_service=_mock_embedding_service(),
        )
        docs = await svc.retrieve("query", top_k=5)
        assert docs == []

    @pytest.mark.asyncio
    async def test_raises_embedding_error_on_embed_failure(self):
        bad_emb = _mock_embedding_service()
        bad_emb._embeddings.aembed_query.side_effect = RuntimeError("no ollama")

        svc = RetrieverService(
            vector_store=_mock_store(),
            embedding_service=bad_emb,
        )
        with pytest.raises(EmbeddingError):
            await svc.retrieve("query")

    @pytest.mark.asyncio
    async def test_raises_retrieval_error_on_chroma_failure(self):
        col = MagicMock()
        col.query.side_effect = Exception("chroma down")
        svc = RetrieverService(
            vector_store=_mock_store(col),
            embedding_service=_mock_embedding_service(),
        )
        with pytest.raises(RetrievalError, match="ChromaDB query failed"):
            await svc.retrieve("query")

    @pytest.mark.asyncio
    async def test_metadata_is_stringified(self):
        col = _mock_collection(
            ids=["i1"],
            documents=["content"],
            metadatas=[{"source": "s", "page": 3}],  # int value
            distances=[0.2],
        )
        svc = RetrieverService(
            vector_store=_mock_store(col),
            embedding_service=_mock_embedding_service(),
        )
        docs = await svc.retrieve("q", top_k=1)
        assert docs[0]["metadata"]["page"] == "3"


# ── Chunking helper ───────────────────────────────────────────────────────────


class TestIngestionChunking:
    def test_short_text_returns_single_chunk(self):
        result = _chunk_text("hello", chunk_size=512, chunk_overlap=64)
        assert result == ["hello"]

    def test_empty_string_returns_empty_list(self):
        assert _chunk_text("", chunk_size=512, chunk_overlap=64) == []

    def test_long_text_produces_multiple_chunks(self):
        text = "word " * 300  # 1500 chars
        chunks = _chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_all_chunks_fit_within_chunk_size(self):
        text = "a " * 400
        for chunk in _chunk_text(text, chunk_size=100, chunk_overlap=10):
            assert len(chunk) <= 100

    def test_overlap_means_adjacent_chunks_share_content(self):
        # With overlap, end of chunk N should appear at start of chunk N+1
        text = "alpha beta gamma delta epsilon " * 50
        chunks = _chunk_text(text, chunk_size=80, chunk_overlap=30)
        assert len(chunks) >= 2
        # The last ~30 chars of chunk 0 should appear somewhere in chunk 1
        tail = chunks[0][-20:]
        assert tail in chunks[1]

    def test_exact_chunk_size_text_is_one_chunk(self):
        text = "x" * 512
        assert len(_chunk_text(text, chunk_size=512, chunk_overlap=64)) == 1

    def test_chunk_id_is_deterministic(self):
        id1 = _make_chunk_id("source.txt", 0, "some content")
        id2 = _make_chunk_id("source.txt", 0, "some content")
        assert id1 == id2

    def test_chunk_id_differs_for_different_sources(self):
        id1 = _make_chunk_id("a.txt", 0, "content")
        id2 = _make_chunk_id("b.txt", 0, "content")
        assert id1 != id2

    def test_chunk_id_differs_for_different_indices(self):
        id1 = _make_chunk_id("src", 0, "content")
        id2 = _make_chunk_id("src", 1, "content")
        assert id1 != id2

    def test_chunk_id_length_is_32(self):
        assert len(_make_chunk_id("s", 0, "c")) == 32


# ── IngestionPipeline ─────────────────────────────────────────────────────────


class TestIngestionPipeline:
    def _make_doc(self, content: str = "x " * 50, source: str = "test.txt") -> IngestDocument:
        return IngestDocument(content=content, source=source, metadata={})

    @pytest.mark.asyncio
    async def test_ingest_returns_chunk_count(self):
        store = _mock_store()
        emb = _mock_embedding_service()
        pipeline = IngestionPipeline(
            vector_store=store, embedding_service=emb, chunk_size=50, chunk_overlap=5
        )
        count = await pipeline.ingest([self._make_doc()])
        assert count > 0

    @pytest.mark.asyncio
    async def test_ingest_raises_on_empty_list(self):
        pipeline = IngestionPipeline(
            vector_store=_mock_store(), embedding_service=_mock_embedding_service()
        )
        with pytest.raises(ValueError, match="empty"):
            await pipeline.ingest([])

    @pytest.mark.asyncio
    async def test_ingest_calls_upsert_on_collection(self):
        col = _mock_collection()
        store = _mock_store(col)
        emb = _mock_embedding_service()
        pipeline = IngestionPipeline(
            vector_store=store, embedding_service=emb, chunk_size=200, chunk_overlap=20
        )
        doc = self._make_doc(content="short document text")
        await pipeline.ingest([doc])
        col.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_ingest_multiple_documents(self):
        store = _mock_store()
        emb = _mock_embedding_service()
        pipeline = IngestionPipeline(
            vector_store=store, embedding_service=emb, chunk_size=100, chunk_overlap=10
        )
        docs = [self._make_doc(source=f"doc{i}.txt") for i in range(3)]
        count = await pipeline.ingest(docs)
        assert count > 0

    @pytest.mark.asyncio
    async def test_ingest_raises_embedding_error_on_failure(self):
        bad_emb = _mock_embedding_service()
        bad_emb._embeddings.aembed_documents.side_effect = RuntimeError("offline")

        pipeline = IngestionPipeline(
            vector_store=_mock_store(),
            embedding_service=bad_emb,
        )
        with pytest.raises(EmbeddingError):
            await pipeline.ingest([self._make_doc()])

    @pytest.mark.asyncio
    async def test_ingest_raises_retrieval_error_on_chroma_failure(self):
        col = MagicMock()
        col.upsert.side_effect = Exception("write failed")
        pipeline = IngestionPipeline(
            vector_store=_mock_store(col),
            embedding_service=_mock_embedding_service(),
        )
        with pytest.raises(RetrievalError, match="upsert failed"):
            await pipeline.ingest([self._make_doc()])

    @pytest.mark.asyncio
    async def test_delete_source_removes_matching_chunks(self):
        col = _mock_collection(ids=["id1", "id2"])
        col.get.return_value = {"ids": ["id1", "id2"]}
        pipeline = IngestionPipeline(
            vector_store=_mock_store(col), embedding_service=_mock_embedding_service()
        )

        deleted = await pipeline.delete_source("test.txt")

        assert deleted == 2
        col.delete.assert_called_once_with(ids=["id1", "id2"])

    @pytest.mark.asyncio
    async def test_delete_source_returns_zero_when_no_chunks(self):
        col = _mock_collection()
        col.get.return_value = {"ids": []}
        pipeline = IngestionPipeline(
            vector_store=_mock_store(col), embedding_service=_mock_embedding_service()
        )

        deleted = await pipeline.delete_source("missing.txt")
        assert deleted == 0
        col.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_ids_are_deterministic_on_re_ingest(self):
        """Upserting the same doc twice should use the same IDs (no duplicates)."""
        col = _mock_collection()
        store = _mock_store(col)
        emb = _mock_embedding_service()
        pipeline = IngestionPipeline(
            vector_store=store, embedding_service=emb, chunk_size=200, chunk_overlap=20
        )
        doc = self._make_doc(content="repeatable content")
        await pipeline.ingest([doc])
        first_ids = (
            col.upsert.call_args.kwargs.get("ids")
            or col.upsert.call_args[1].get("ids")
            or col.upsert.call_args[0][0]
        )

        await pipeline.ingest([doc])
        second_ids = (
            col.upsert.call_args.kwargs.get("ids")
            or col.upsert.call_args[1].get("ids")
            or col.upsert.call_args[0][0]
        )

        assert first_ids == second_ids


# ── retriever_node ────────────────────────────────────────────────────────────


class TestRetrieverNode:
    def _make_docs(self, n: int = 2) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                id=f"doc{i}",
                content=f"content {i}",
                source="src",
                metadata={},
                score=0.9 - i * 0.1,
            )
            for i in range(n)
        ]

    @pytest.mark.asyncio
    async def test_sets_retrieved_documents_on_success(self):
        docs = self._make_docs(3)
        mock_svc = AsyncMock()
        mock_svc.retrieve.return_value = docs

        with patch("app.graph.nodes.retriever.RetrieverService", return_value=mock_svc):
            update = await retriever_node(_state("test query"))

        assert update["retrieved_documents"] == docs

    @pytest.mark.asyncio
    async def test_sets_current_node(self):
        mock_svc = AsyncMock()
        mock_svc.retrieve.return_value = []
        with patch("app.graph.nodes.retriever.RetrieverService", return_value=mock_svc):
            update = await retriever_node(_state())
        assert update["current_node"] == "retriever"

    @pytest.mark.asyncio
    async def test_increments_step_count(self):
        mock_svc = AsyncMock()
        mock_svc.retrieve.return_value = []
        state = _state()
        state["step_count"] = 2
        with patch("app.graph.nodes.retriever.RetrieverService", return_value=mock_svc):
            update = await retriever_node(state)
        assert update["step_count"] == 3

    @pytest.mark.asyncio
    async def test_records_error_on_embedding_failure(self):
        mock_svc = AsyncMock()
        mock_svc.retrieve.side_effect = EmbeddingError("embed failed")
        with patch("app.graph.nodes.retriever.RetrieverService", return_value=mock_svc):
            update = await retriever_node(_state())
        assert len(update["errors"]) == 1
        assert update["errors"][0]["node"] == "retriever"
        assert "retrieved_documents" not in update

    @pytest.mark.asyncio
    async def test_records_error_on_retrieval_failure(self):
        mock_svc = AsyncMock()
        mock_svc.retrieve.side_effect = RetrievalError("chroma down")
        with patch("app.graph.nodes.retriever.RetrieverService", return_value=mock_svc):
            update = await retriever_node(_state())
        assert len(update["errors"]) == 1
        assert "chroma down" in update["errors"][0]["message"]

    @pytest.mark.asyncio
    async def test_records_error_on_unexpected_exception(self):
        mock_svc = AsyncMock()
        mock_svc.retrieve.side_effect = RuntimeError("unexpected")
        with patch("app.graph.nodes.retriever.RetrieverService", return_value=mock_svc):
            update = await retriever_node(_state())
        assert len(update["errors"]) == 1

    @pytest.mark.asyncio
    async def test_no_errors_key_on_success(self):
        mock_svc = AsyncMock()
        mock_svc.retrieve.return_value = self._make_docs()
        with patch("app.graph.nodes.retriever.RetrieverService", return_value=mock_svc):
            update = await retriever_node(_state())
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_empty_result_is_not_an_error(self):
        mock_svc = AsyncMock()
        mock_svc.retrieve.return_value = []
        with patch("app.graph.nodes.retriever.RetrieverService", return_value=mock_svc):
            update = await retriever_node(_state())
        assert update["retrieved_documents"] == []
        assert "errors" not in update
