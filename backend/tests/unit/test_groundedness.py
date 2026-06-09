"""
Unit tests for the groundedness evaluation pipeline.

All tests are pure (no I/O, no LLM calls): the GroundednessEvaluator is
injected with a mock that returns pre-built GroundednessEvaluation objects.

Test groups
-----------
TestBuildGroundednessResult  — score computation, claim partitioning, timestamp
TestEvaluateGroundedness     — service-layer orchestration, evaluator injection
TestGroundednessNode         — graph node state transitions and error handling
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.evaluation.schemas import ClaimVerdict, GroundednessEvaluation
from app.evaluation.service import build_groundedness_result, evaluate_groundedness
from app.graph.nodes.groundedness import groundedness_node
from app.graph.state import RankedDocument, StructuredOutput, initial_state


# ── Factories ──────────────────────────────────────────────────────────────────


def _verdict(claim: str, supported: bool, doc_ids: list[str] | None = None) -> ClaimVerdict:
    return ClaimVerdict(
        claim=claim,
        supported=supported,
        source_document_ids=doc_ids or (["d1"] if supported else []),
        reasoning="test reasoning",
    )


def _evaluation(*verdicts: ClaimVerdict) -> GroundednessEvaluation:
    return GroundednessEvaluation(
        claims=list(verdicts),
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


def _structured_output(answer: str = "The answer is X.") -> StructuredOutput:
    return StructuredOutput(summary="summary", answer=answer, citations=[])


def _mock_evaluator(evaluation: GroundednessEvaluation) -> MagicMock:
    ev = MagicMock()
    ev.evaluate = AsyncMock(return_value=evaluation)
    return ev


# ── build_groundedness_result ──────────────────────────────────────────────────


class TestBuildGroundednessResult:
    def test_all_supported_score_is_one(self):
        ev = _evaluation(_verdict("c1", True), _verdict("c2", True))
        result = build_groundedness_result(ev)
        assert result["groundedness_score"] == pytest.approx(1.0)

    def test_all_unsupported_score_is_zero(self):
        ev = _evaluation(_verdict("c1", False), _verdict("c2", False))
        result = build_groundedness_result(ev)
        assert result["groundedness_score"] == pytest.approx(0.0)

    def test_empty_claims_score_is_zero(self):
        ev = _evaluation()
        result = build_groundedness_result(ev)
        assert result["groundedness_score"] == 0.0

    def test_half_supported(self):
        ev = _evaluation(_verdict("c1", True), _verdict("c2", False))
        result = build_groundedness_result(ev)
        assert result["groundedness_score"] == pytest.approx(0.5)

    def test_one_of_three_supported(self):
        ev = _evaluation(
            _verdict("c1", True),
            _verdict("c2", False),
            _verdict("c3", False),
        )
        result = build_groundedness_result(ev)
        assert result["groundedness_score"] == pytest.approx(round(1 / 3, 4), abs=1e-4)

    def test_claims_partitioned_correctly(self):
        ev = _evaluation(_verdict("yes", True), _verdict("no", False))
        result = build_groundedness_result(ev)
        assert len(result["supported_claims"]) == 1
        assert len(result["unsupported_claims"]) == 1
        assert result["supported_claims"][0]["claim"] == "yes"
        assert result["unsupported_claims"][0]["claim"] == "no"

    def test_evaluated_at_is_iso_string(self):
        ev = _evaluation(_verdict("c", True))
        result = build_groundedness_result(ev)
        # Must be parseable ISO-8601 — just check it's a non-empty string with T
        assert "T" in result["evaluated_at"]

    def test_score_rounded_to_four_decimal_places(self):
        # 2/3 = 0.6666... → rounded to 0.6667
        ev = _evaluation(_verdict("c1", True), _verdict("c2", True), _verdict("c3", False))
        result = build_groundedness_result(ev)
        assert result["groundedness_score"] == pytest.approx(0.6667, abs=1e-4)

    def test_source_document_ids_preserved(self):
        ev = _evaluation(_verdict("c1", True, doc_ids=["doc-a", "doc-b"]))
        result = build_groundedness_result(ev)
        assert result["supported_claims"][0]["source_document_ids"] == ["doc-a", "doc-b"]


# ── evaluate_groundedness (service) ───────────────────────────────────────────


class TestEvaluateGroundedness:
    @pytest.mark.asyncio
    async def test_delegates_to_evaluator(self):
        ev = _evaluation(_verdict("c1", True), _verdict("c2", False))
        mock_ev = _mock_evaluator(ev)

        result = await evaluate_groundedness(
            query="What is X?",
            answer="X is Y.",
            documents=[_ranked()],
            evaluator=mock_ev,
        )

        mock_ev.evaluate.assert_awaited_once()
        assert result["groundedness_score"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_returns_groundedness_result_typeddict(self):
        ev = _evaluation(_verdict("c1", True))
        result = await evaluate_groundedness(
            query="q", answer="a", documents=[], evaluator=_mock_evaluator(ev)
        )
        assert "groundedness_score" in result
        assert "supported_claims" in result
        assert "unsupported_claims" in result
        assert "evaluated_at" in result

    @pytest.mark.asyncio
    async def test_propagates_evaluator_exception(self):
        mock_ev = MagicMock()
        mock_ev.evaluate = AsyncMock(side_effect=RuntimeError("LLM down"))

        with pytest.raises(RuntimeError, match="LLM down"):
            await evaluate_groundedness(
                query="q", answer="a", documents=[], evaluator=mock_ev
            )


# ── groundedness_node ─────────────────────────────────────────────────────────


class TestGroundednessNode:
    def _state(self, answer: str | None = "X is Y.") -> dict:
        s = dict(initial_state(session_id="s1", query="What is X?"))
        if answer is not None:
            s["structured_output"] = _structured_output(answer)
            s["reranked_documents"] = [_ranked()]
        return s

    @pytest.mark.asyncio
    async def test_successful_evaluation_writes_groundedness(self, monkeypatch):
        ev = _evaluation(_verdict("c1", True), _verdict("c2", True))

        async def _fake_evaluate(query, answer, documents, evaluator=None):
            return build_groundedness_result(ev)

        monkeypatch.setattr(
            "app.graph.nodes.groundedness.evaluate_groundedness", _fake_evaluate
        )

        state = self._state()
        result = await groundedness_node(state)

        assert "groundedness" in result
        assert result["groundedness"]["groundedness_score"] == pytest.approx(1.0)
        assert result["current_node"] == "groundedness"
        assert result["step_count"] == 1

    @pytest.mark.asyncio
    async def test_no_structured_output_skips_evaluation(self):
        state = self._state(answer=None)
        result = await groundedness_node(state)

        assert "groundedness" not in result
        assert "errors" not in result
        assert result["current_node"] == "groundedness"

    @pytest.mark.asyncio
    async def test_empty_answer_skips_evaluation(self):
        state = self._state(answer="")
        result = await groundedness_node(state)

        assert "groundedness" not in result
        assert "errors" not in result

    @pytest.mark.asyncio
    async def test_evaluator_failure_appends_error_and_continues(self, monkeypatch):
        async def _fail(query, answer, documents, evaluator=None):
            raise RuntimeError("timeout")

        monkeypatch.setattr(
            "app.graph.nodes.groundedness.evaluate_groundedness", _fail
        )

        state = self._state()
        result = await groundedness_node(state)

        assert "groundedness" not in result
        assert len(result["errors"]) == 1
        assert "timeout" in result["errors"][0]["message"]
        assert result["current_node"] == "groundedness"

    @pytest.mark.asyncio
    async def test_step_count_incremented(self, monkeypatch):
        ev = _evaluation(_verdict("c1", True))

        async def _fake_evaluate(query, answer, documents, evaluator=None):
            return build_groundedness_result(ev)

        monkeypatch.setattr(
            "app.graph.nodes.groundedness.evaluate_groundedness", _fake_evaluate
        )

        state = self._state()
        state["step_count"] = 7
        result = await groundedness_node(state)

        assert result["step_count"] == 8
