"""
Unit tests for the reranking pipeline.

No model is downloaded — every test injects a mock CrossEncoder so the
suite runs in milliseconds without any GPU or network dependency.

Test groups
-----------
TestSigmoid            — _sigmoid helper: boundaries, monotonicity, stability
TestToRanked           — _to_ranked: field mapping, score passthrough
TestRerankerService    — rerank(): top-n slicing, ordering, score normalization,
                         empty input, error handling, model injection
TestRerankerNode       — node state contract: writes, step_count, error recording
"""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.core.exceptions import RerankingError
from app.graph.nodes.reranker import reranker_node
from app.graph.state import AppState, RankedDocument, RetrievedDocument, initial_state
from app.rag.reranker import RerankerService, _sigmoid, _to_ranked

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_doc(doc_id: str = "d1", content: str = "text", score: float = 0.8) -> RetrievedDocument:
    return RetrievedDocument(id=doc_id, content=content, source="src", metadata={}, score=score)


def _make_docs(n: int, base_score: float = 0.9) -> list[RetrievedDocument]:
    return [_make_doc(f"d{i}", f"content {i}", base_score - i * 0.05) for i in range(n)]


def _mock_cross_encoder(logits: list[float]) -> MagicMock:
    """Mock whose predict() returns a numpy array of *logits*."""
    m = MagicMock()
    m.predict.return_value = np.array(logits, dtype=np.float32)
    return m


def _state(query: str = "test query") -> AppState:
    return initial_state(session_id="sess-r", query=query)


# ── _sigmoid ──────────────────────────────────────────────────────────────────


class TestSigmoid:
    def test_zero_maps_to_half(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-9)

    def test_large_negative_approaches_zero(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-9)

    def test_output_is_strictly_between_zero_and_one(self):
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            s = _sigmoid(x)
            assert 0.0 < s < 1.0

    def test_monotonically_increasing(self):
        prev = _sigmoid(-5.0)
        for x in [-4.0, -2.0, 0.0, 2.0, 4.0]:
            curr = _sigmoid(x)
            assert curr > prev
            prev = curr

    def test_symmetry_around_zero(self):
        assert _sigmoid(2.0) + _sigmoid(-2.0) == pytest.approx(1.0)

    def test_known_value(self):
        # sigmoid(1) = e / (1 + e)
        expected = math.e / (1.0 + math.e)
        assert _sigmoid(1.0) == pytest.approx(expected, rel=1e-6)


# ── _to_ranked ────────────────────────────────────────────────────────────────


class TestToRanked:
    def test_preserves_id(self):
        doc = _make_doc(doc_id="abc")
        assert _to_ranked(doc, 0.7)["id"] == "abc"

    def test_preserves_content(self):
        doc = _make_doc(content="hello world")
        assert _to_ranked(doc, 0.5)["content"] == "hello world"

    def test_preserves_source(self):
        doc = RetrievedDocument(id="x", content="c", source="wiki.txt", metadata={}, score=0.6)
        assert _to_ranked(doc, 0.9)["source"] == "wiki.txt"

    def test_preserves_metadata(self):
        meta = {"page": "3", "lang": "en"}
        doc = RetrievedDocument(id="x", content="c", source="s", metadata=meta, score=0.5)
        assert _to_ranked(doc, 0.8)["metadata"] == meta

    def test_sets_retrieval_score_from_doc_score(self):
        doc = _make_doc(score=0.72)
        ranked = _to_ranked(doc, 0.9)
        assert ranked["retrieval_score"] == pytest.approx(0.72)

    def test_sets_rerank_score(self):
        doc = _make_doc()
        assert _to_ranked(doc, 0.65)["rerank_score"] == pytest.approx(0.65)

    def test_all_fields_present(self):
        ranked = _to_ranked(_make_doc(), 0.5)
        assert set(ranked.keys()) == {
            "id",
            "content",
            "source",
            "metadata",
            "retrieval_score",
            "rerank_score",
        }


# ── RerankerService ───────────────────────────────────────────────────────────


class TestRerankerService:
    def setup_method(self) -> None:
        RerankerService._reset_model_cache()

    @pytest.mark.asyncio
    async def test_returns_top_n_documents(self):
        docs = _make_docs(10)
        logits = [float(i) for i in range(10)]  # 0..9
        svc = RerankerService(cross_encoder=_mock_cross_encoder(logits))
        result = await svc.rerank("query", docs, top_n=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_returns_default_top_n_from_settings(self):
        svc = RerankerService(cross_encoder=_mock_cross_encoder([1.0, 2.0, 3.0, 0.0]))
        docs = _make_docs(4)
        result = await svc.rerank("query", docs)
        assert len(result) == svc._settings.rag.reranker_top_n

    @pytest.mark.asyncio
    async def test_ordered_by_rerank_score_descending(self):
        docs = _make_docs(5)
        # logits: doc2 has highest, doc4 second, doc0 third
        logits = [1.0, 0.5, 5.0, -1.0, 3.0]
        svc = RerankerService(cross_encoder=_mock_cross_encoder(logits))
        result = await svc.rerank("query", docs, top_n=3)
        assert result[0]["id"] == "d2"
        assert result[1]["id"] == "d4"
        assert result[2]["id"] == "d0"

    @pytest.mark.asyncio
    async def test_scores_are_sigmoid_normalised(self):
        docs = _make_docs(3)
        logits = [0.0, 2.0, -2.0]
        svc = RerankerService(cross_encoder=_mock_cross_encoder(logits))
        result = await svc.rerank("query", docs, top_n=3)
        for doc in result:
            assert 0.0 < doc["rerank_score"] < 1.0

    @pytest.mark.asyncio
    async def test_higher_logit_yields_higher_score(self):
        docs = _make_docs(2)
        logits = [-3.0, 3.0]
        svc = RerankerService(cross_encoder=_mock_cross_encoder(logits))
        result = await svc.rerank("query", docs, top_n=2)
        assert result[0]["rerank_score"] > result[1]["rerank_score"]

    @pytest.mark.asyncio
    async def test_preserves_retrieval_score_on_each_document(self):
        doc = _make_doc(score=0.77)
        svc = RerankerService(cross_encoder=_mock_cross_encoder([1.0]))
        result = await svc.rerank("query", [doc], top_n=1)
        assert result[0]["retrieval_score"] == pytest.approx(0.77)

    @pytest.mark.asyncio
    async def test_caps_top_n_when_fewer_documents_available(self):
        docs = _make_docs(2)
        svc = RerankerService(cross_encoder=_mock_cross_encoder([1.0, 0.5]))
        result = await svc.rerank("query", docs, top_n=10)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_empty_documents(self):
        svc = RerankerService(cross_encoder=_mock_cross_encoder([]))
        result = await svc.rerank("query", [], top_n=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_value_error_for_blank_query(self):
        svc = RerankerService(cross_encoder=_mock_cross_encoder([1.0]))
        with pytest.raises(ValueError, match="blank"):
            await svc.rerank("   ", _make_docs(1), top_n=1)

    @pytest.mark.asyncio
    async def test_raises_reranking_error_when_predict_fails(self):
        bad_model = MagicMock()
        bad_model.predict.side_effect = RuntimeError("CUDA OOM")
        svc = RerankerService(cross_encoder=bad_model)
        with pytest.raises(RerankingError, match="CrossEncoder.predict failed"):
            await svc.rerank("query", _make_docs(3), top_n=2)

    @pytest.mark.asyncio
    async def test_injected_model_never_loads_from_cache(self):
        """Class-level cache must be untouched when a model is injected."""
        mock = _mock_cross_encoder([1.0])
        svc = RerankerService(cross_encoder=mock)
        await svc.rerank("q", _make_docs(1), top_n=1)
        assert RerankerService._cached_model is None

    @pytest.mark.asyncio
    async def test_numpy_array_scores_convert_to_python_floats(self):
        docs = _make_docs(2)
        # numpy float32 values
        model = MagicMock()
        model.predict.return_value = np.array([1.5, -0.5], dtype=np.float32)
        svc = RerankerService(cross_encoder=model)
        result = await svc.rerank("query", docs, top_n=2)
        for doc in result:
            assert isinstance(doc["rerank_score"], float)

    @pytest.mark.asyncio
    async def test_top_one_returns_single_best_document(self):
        docs = _make_docs(5)
        # doc index 3 has the highest logit
        logits = [0.1, 0.2, 0.3, 9.9, 0.5]
        svc = RerankerService(cross_encoder=_mock_cross_encoder(logits))
        result = await svc.rerank("query", docs, top_n=1)
        assert len(result) == 1
        assert result[0]["id"] == "d3"

    @pytest.mark.asyncio
    async def test_ranked_document_has_all_required_fields(self):
        doc = _make_doc()
        svc = RerankerService(cross_encoder=_mock_cross_encoder([0.8]))
        result = await svc.rerank("query", [doc], top_n=1)
        keys = set(result[0].keys())
        assert keys == {"id", "content", "source", "metadata", "retrieval_score", "rerank_score"}


# ── reranker_node ─────────────────────────────────────────────────────────────


class TestRerankerNode:
    def _ranked_doc(self, doc_id: str = "r1", score: float = 0.9) -> RankedDocument:
        return RankedDocument(
            id=doc_id,
            content="text",
            source="s",
            metadata={},
            retrieval_score=0.8,
            rerank_score=score,
        )

    def _state_with_docs(self, n: int = 5) -> AppState:
        state = _state()
        state["retrieved_documents"] = _make_docs(n)
        return state

    @pytest.mark.asyncio
    async def test_sets_reranked_documents_on_success(self):
        ranked = [self._ranked_doc(f"r{i}") for i in range(3)]
        mock_svc = AsyncMock()
        mock_svc.rerank.return_value = ranked
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            update = await reranker_node(self._state_with_docs())
        assert update["reranked_documents"] == ranked

    @pytest.mark.asyncio
    async def test_sets_current_node(self):
        mock_svc = AsyncMock()
        mock_svc.rerank.return_value = []
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            update = await reranker_node(self._state_with_docs())
        assert update["current_node"] == "reranker"

    @pytest.mark.asyncio
    async def test_increments_step_count_from_zero(self):
        mock_svc = AsyncMock()
        mock_svc.rerank.return_value = []
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            update = await reranker_node(_state_with_step(0))
        assert update["step_count"] == 1

    @pytest.mark.asyncio
    async def test_increments_step_count_from_existing(self):
        mock_svc = AsyncMock()
        mock_svc.rerank.return_value = []
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            update = await reranker_node(_state_with_step(7))
        assert update["step_count"] == 8

    @pytest.mark.asyncio
    async def test_passes_query_and_retrieved_docs_to_service(self):
        mock_svc = AsyncMock()
        mock_svc.rerank.return_value = []
        docs = _make_docs(4)
        state = _state("find me something")
        state["retrieved_documents"] = docs
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            await reranker_node(state)
        mock_svc.rerank.assert_called_once()
        call_kwargs = mock_svc.rerank.call_args
        assert call_kwargs.kwargs["query"] == "find me something"
        assert call_kwargs.kwargs["documents"] == docs

    @pytest.mark.asyncio
    async def test_records_error_on_reranking_failure(self):
        mock_svc = AsyncMock()
        mock_svc.rerank.side_effect = RerankingError("model exploded")
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            update = await reranker_node(self._state_with_docs())
        assert len(update["errors"]) == 1
        assert update["errors"][0]["node"] == "reranker"
        assert "model exploded" in update["errors"][0]["message"]
        assert "reranked_documents" not in update

    @pytest.mark.asyncio
    async def test_records_error_on_value_error(self):
        mock_svc = AsyncMock()
        mock_svc.rerank.side_effect = ValueError("blank query")
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            update = await reranker_node(self._state_with_docs())
        assert len(update["errors"]) == 1

    @pytest.mark.asyncio
    async def test_records_error_on_unexpected_exception(self):
        mock_svc = AsyncMock()
        mock_svc.rerank.side_effect = MemoryError("OOM")
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            update = await reranker_node(self._state_with_docs())
        assert len(update["errors"]) == 1

    @pytest.mark.asyncio
    async def test_no_errors_key_on_success(self):
        mock_svc = AsyncMock()
        mock_svc.rerank.return_value = [self._ranked_doc()]
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            update = await reranker_node(self._state_with_docs())
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_empty_retrieved_documents_returns_empty_reranked(self):
        mock_svc = AsyncMock()
        mock_svc.rerank.return_value = []
        state = _state()
        state["retrieved_documents"] = []
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            update = await reranker_node(state)
        assert update["reranked_documents"] == []
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_missing_retrieved_documents_treated_as_empty(self):
        """Node must not crash when retrieved_documents is absent from state."""
        mock_svc = AsyncMock()
        mock_svc.rerank.return_value = []
        # _state() does not set retrieved_documents
        bare = _state()
        with patch("app.graph.nodes.reranker.RerankerService", return_value=mock_svc):
            update = await reranker_node(bare)
        assert update["reranked_documents"] == []


# ── Helpers used in TestRerankerNode ──────────────────────────────────────────


def _state_with_step(step: int) -> AppState:
    s = _state()
    s["step_count"] = step
    return s
