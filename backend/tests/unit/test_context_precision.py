"""
Unit tests for the context precision evaluation pipeline.

All tests are pure (no I/O, no LLM calls): the ContextPrecisionEvaluator is
injected via monkeypatching or a mock that returns pre-built objects.

Test groups
-----------
TestContextEvalNode  — graph node state transitions, score computation, error handling
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.evaluation.context_precision_schemas import (
    ContextPrecisionEvaluation,
    DocumentRelevanceVerdict,
)
from app.graph.nodes.context_eval import context_eval_node
from app.graph.state import RankedDocument, initial_state

# ── Factories ──────────────────────────────────────────────────────────────────


def _verdict(doc_id: str, is_relevant: bool) -> DocumentRelevanceVerdict:
    return DocumentRelevanceVerdict(
        document_id=doc_id,
        is_relevant=is_relevant,
        reasoning="test reasoning",
    )


def _evaluation(*verdicts: DocumentRelevanceVerdict) -> ContextPrecisionEvaluation:
    return ContextPrecisionEvaluation(
        verdicts=list(verdicts),
        overall_reasoning="test overall reasoning",
    )


def _ranked(doc_id: str = "d1") -> RankedDocument:
    return RankedDocument(
        id=doc_id,
        content="relevant content",
        source="test_source",
        metadata={},
        retrieval_score=0.8,
        rerank_score=0.9,
    )


def _mock_evaluator(evaluation: ContextPrecisionEvaluation) -> MagicMock:
    ev = MagicMock()
    ev.evaluate = AsyncMock(return_value=evaluation)
    return ev


# ── context_eval_node ─────────────────────────────────────────────────────────


class TestContextEvalNode:
    def _state(self, docs: list[RankedDocument] | None = None) -> dict:
        s = dict(initial_state(session_id="s1", query="What is X?"))
        s["reranked_documents"] = docs if docs is not None else [_ranked()]
        return s

    @pytest.mark.asyncio
    async def test_all_relevant_score_is_one(self, monkeypatch):
        ev = _evaluation(_verdict("d1", True), _verdict("d2", True))

        monkeypatch.setattr(
            "app.graph.nodes.context_eval.ContextPrecisionEvaluator",
            lambda **_: _mock_evaluator(ev),
        )
        monkeypatch.setattr("app.graph.nodes.context_eval.instrumented_llm", lambda _: None)

        state = self._state([_ranked("d1"), _ranked("d2")])
        result = await context_eval_node(state)

        assert "context_precision" in result
        assert result["context_precision"]["context_precision_score"] == pytest.approx(1.0)
        assert len(result["context_precision"]["relevant_documents"]) == 2
        assert len(result["context_precision"]["irrelevant_documents"]) == 0

    @pytest.mark.asyncio
    async def test_all_irrelevant_score_is_zero(self, monkeypatch):
        ev = _evaluation(_verdict("d1", False), _verdict("d2", False))

        monkeypatch.setattr(
            "app.graph.nodes.context_eval.ContextPrecisionEvaluator",
            lambda **_: _mock_evaluator(ev),
        )
        monkeypatch.setattr("app.graph.nodes.context_eval.instrumented_llm", lambda _: None)

        state = self._state([_ranked("d1"), _ranked("d2")])
        result = await context_eval_node(state)

        assert result["context_precision"]["context_precision_score"] == pytest.approx(0.0)
        assert len(result["context_precision"]["irrelevant_documents"]) == 2

    @pytest.mark.asyncio
    async def test_half_relevant_score_is_half(self, monkeypatch):
        ev = _evaluation(_verdict("d1", True), _verdict("d2", False))

        monkeypatch.setattr(
            "app.graph.nodes.context_eval.ContextPrecisionEvaluator",
            lambda **_: _mock_evaluator(ev),
        )
        monkeypatch.setattr("app.graph.nodes.context_eval.instrumented_llm", lambda _: None)

        state = self._state([_ranked("d1"), _ranked("d2")])
        result = await context_eval_node(state)

        assert result["context_precision"]["context_precision_score"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_no_documents_skips_evaluation(self):
        state = self._state(docs=[])
        result = await context_eval_node(state)

        assert "context_precision" not in result
        assert "errors" not in result
        assert result["current_node"] == "context_eval"

    @pytest.mark.asyncio
    async def test_evaluator_failure_appends_error_and_continues(self, monkeypatch):
        mock_ev = MagicMock()
        mock_ev.evaluate = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        monkeypatch.setattr(
            "app.graph.nodes.context_eval.ContextPrecisionEvaluator",
            lambda **_: mock_ev,
        )
        monkeypatch.setattr("app.graph.nodes.context_eval.instrumented_llm", lambda _: None)

        state = self._state()
        result = await context_eval_node(state)

        assert "context_precision" not in result
        assert len(result["errors"]) == 1
        assert "LLM timeout" in result["errors"][0]["message"]
        assert result["current_node"] == "context_eval"

    @pytest.mark.asyncio
    async def test_step_count_incremented(self, monkeypatch):
        ev = _evaluation(_verdict("d1", True))

        monkeypatch.setattr(
            "app.graph.nodes.context_eval.ContextPrecisionEvaluator",
            lambda **_: _mock_evaluator(ev),
        )
        monkeypatch.setattr("app.graph.nodes.context_eval.instrumented_llm", lambda _: None)

        state = self._state()
        state["step_count"] = 5
        result = await context_eval_node(state)

        assert result["step_count"] == 6

    @pytest.mark.asyncio
    async def test_result_contains_evaluated_at(self, monkeypatch):
        ev = _evaluation(_verdict("d1", True))

        monkeypatch.setattr(
            "app.graph.nodes.context_eval.ContextPrecisionEvaluator",
            lambda **_: _mock_evaluator(ev),
        )
        monkeypatch.setattr("app.graph.nodes.context_eval.instrumented_llm", lambda _: None)

        state = self._state()
        result = await context_eval_node(state)

        assert "T" in result["context_precision"]["evaluated_at"]

    @pytest.mark.asyncio
    async def test_verdicts_partitioned_correctly(self, monkeypatch):
        ev = _evaluation(
            _verdict("d1", True),
            _verdict("d2", False),
            _verdict("d3", True),
        )

        monkeypatch.setattr(
            "app.graph.nodes.context_eval.ContextPrecisionEvaluator",
            lambda **_: _mock_evaluator(ev),
        )
        monkeypatch.setattr("app.graph.nodes.context_eval.instrumented_llm", lambda _: None)

        state = self._state([_ranked("d1"), _ranked("d2"), _ranked("d3")])
        result = await context_eval_node(state)

        relevant_ids = {v["document_id"] for v in result["context_precision"]["relevant_documents"]}
        irrelevant_ids = {
            v["document_id"] for v in result["context_precision"]["irrelevant_documents"]
        }
        assert relevant_ids == {"d1", "d3"}
        assert irrelevant_ids == {"d2"}
        assert result["context_precision"]["context_precision_score"] == pytest.approx(
            round(2 / 3, 4), abs=1e-4
        )
