"""
Unit tests for the LLM-as-a-judge feature.

Test groups
-----------
TestComputeOverallScore    — deterministic weighted-average formula
TestLLMJudgeNode           — node read/write contract, skip conditions, failure handling
TestAutoApprovalGateVeto   — veto gate: both confidence AND judge must pass
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.evaluation.judge import compute_overall_score
from app.evaluation.judge_schemas import DimensionScore, JudgeEvaluation
from app.graph.nodes.auto_approval import auto_approval_gate_node
from app.graph.nodes.llm_judge import llm_judge_node
from app.graph.state import AppState, JudgeDimensionScore, JudgeResult

# ── Factories ──────────────────────────────────────────────────────────────────


def _dim(score: float, reasoning: str = "ok") -> DimensionScore:
    return DimensionScore(score=score, reasoning=reasoning)


def _evaluation(
    faithfulness: float = 0.9,
    relevance: float = 0.9,
    completeness: float = 0.8,
    coherence: float = 0.8,
    critique: str = "Good answer.",
) -> JudgeEvaluation:
    return JudgeEvaluation(
        faithfulness=_dim(faithfulness),
        relevance=_dim(relevance),
        completeness=_dim(completeness),
        coherence=_dim(coherence),
        critique=critique,
    )


def _state(
    answer: str = "The answer is 42.",
    route: str = "research",
    judge_result: JudgeResult | None = None,
    router_conf: float = 0.9,
    retrieval_conf: float = 0.8,
    answer_conf: float = 0.85,
) -> AppState:
    so = {"summary": "s", "answer": answer, "citations": []} if answer else None
    state: AppState = {
        "session_id": "test-session",
        "query": "What is the answer?",
        "route": route,  # type: ignore[typeddict-item]
        "structured_output": so,
        "reranked_documents": [],
        "errors": [],
        "step_count": 0,
        "router_confidence": router_conf,
        "retrieval_confidence": retrieval_conf,
        "answer_confidence": answer_conf,
    }
    if judge_result is not None:
        state["judge_result"] = judge_result
    return state


def _judge_result(overall: float = 0.85, recommendation: str = "auto_approve") -> JudgeResult:
    dim = JudgeDimensionScore(score=overall, reasoning="ok")
    return JudgeResult(
        faithfulness=dim,
        relevance=dim,
        completeness=dim,
        coherence=dim,
        overall_score=overall,
        recommendation=recommendation,
        critique="Good.",
        evaluated_at="2026-01-01T00:00:00+00:00",
    )


# ── TestComputeOverallScore ────────────────────────────────────────────────────


class TestComputeOverallScore:
    def test_all_perfect_returns_one(self):
        ev = _evaluation(1.0, 1.0, 1.0, 1.0)
        assert compute_overall_score(ev) == pytest.approx(1.0)

    def test_all_zero_returns_zero(self):
        ev = _evaluation(0.0, 0.0, 0.0, 0.0)
        assert compute_overall_score(ev) == pytest.approx(0.0)

    def test_weights_faithfulness_dominates(self):
        # Only faithfulness = 1.0 → weight 0.40
        faith_only = compute_overall_score(_evaluation(1.0, 0.0, 0.0, 0.0))
        rel_only = compute_overall_score(_evaluation(0.0, 1.0, 0.0, 0.0))
        comp_only = compute_overall_score(_evaluation(0.0, 0.0, 1.0, 0.0))
        coh_only = compute_overall_score(_evaluation(0.0, 0.0, 0.0, 1.0))
        assert faith_only > rel_only > comp_only > coh_only

    def test_individual_weights_exact(self):
        assert compute_overall_score(_evaluation(1.0, 0.0, 0.0, 0.0)) == pytest.approx(
            0.40, abs=1e-4
        )
        assert compute_overall_score(_evaluation(0.0, 1.0, 0.0, 0.0)) == pytest.approx(
            0.30, abs=1e-4
        )
        assert compute_overall_score(_evaluation(0.0, 0.0, 1.0, 0.0)) == pytest.approx(
            0.20, abs=1e-4
        )
        assert compute_overall_score(_evaluation(0.0, 0.0, 0.0, 1.0)) == pytest.approx(
            0.10, abs=1e-4
        )

    def test_weights_sum_to_one_on_uniform_input(self):
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert compute_overall_score(_evaluation(v, v, v, v)) == pytest.approx(v, abs=1e-4)

    def test_spot_check(self):
        # 0.40*0.8 + 0.30*0.7 + 0.20*0.6 + 0.10*0.5 = 0.32+0.21+0.12+0.05 = 0.70
        ev = _evaluation(faithfulness=0.8, relevance=0.7, completeness=0.6, coherence=0.5)
        assert compute_overall_score(ev) == pytest.approx(0.70, abs=1e-4)

    def test_result_clamped_to_unit_interval(self):
        # DimensionScore enforces [0,1] via Pydantic, but compute_overall_score
        # also clamps defensively; verify it returns in [0.0, 1.0].
        result = compute_overall_score(_evaluation(0.5, 0.5, 0.5, 0.5))
        assert 0.0 <= result <= 1.0


# ── TestLLMJudgeNode ──────────────────────────────────────────────────────────


class TestLLMJudgeNode:
    @pytest.mark.asyncio
    async def test_skip_when_no_structured_output(self):
        state = _state()
        state["structured_output"] = None
        result = await llm_judge_node(state)
        assert "judge_result" not in result
        assert result["step_count"] == 1

    @pytest.mark.asyncio
    async def test_skip_when_empty_answer(self):
        state = _state(answer="")
        result = await llm_judge_node(state)
        assert "judge_result" not in result

    @pytest.mark.asyncio
    async def test_returns_judge_result_on_success(self):
        ev = _evaluation(0.9, 0.85, 0.8, 0.75, critique="Solid answer.")
        mock_judge = AsyncMock()
        mock_judge.evaluate.return_value = (ev, compute_overall_score(ev))

        with patch("app.graph.nodes.llm_judge.LLMJudge", return_value=mock_judge):
            result = await llm_judge_node(_state())

        assert "judge_result" in result
        jr = result["judge_result"]
        assert jr["faithfulness"]["score"] == pytest.approx(0.9)
        assert jr["relevance"]["score"] == pytest.approx(0.85)
        assert jr["completeness"]["score"] == pytest.approx(0.8)
        assert jr["coherence"]["score"] == pytest.approx(0.75)
        assert 0.0 <= jr["overall_score"] <= 1.0
        assert jr["recommendation"] in ("auto_approve", "needs_review")
        assert "evaluated_at" in jr
        assert result["step_count"] == 1

    @pytest.mark.asyncio
    async def test_recommendation_auto_approve_above_threshold(self):
        ev = _evaluation(1.0, 1.0, 1.0, 1.0)
        mock_judge = AsyncMock()
        mock_judge.evaluate.return_value = (ev, 1.0)

        with patch("app.graph.nodes.llm_judge.LLMJudge", return_value=mock_judge):
            result = await llm_judge_node(_state())

        assert result["judge_result"]["recommendation"] == "auto_approve"

    @pytest.mark.asyncio
    async def test_recommendation_needs_review_below_threshold(self):
        ev = _evaluation(0.4, 0.3, 0.2, 0.1)
        mock_judge = AsyncMock()
        mock_judge.evaluate.return_value = (ev, compute_overall_score(ev))

        with patch("app.graph.nodes.llm_judge.LLMJudge", return_value=mock_judge):
            result = await llm_judge_node(_state())

        assert result["judge_result"]["recommendation"] == "needs_review"

    @pytest.mark.asyncio
    async def test_failure_appends_error_and_skips_judge_result(self):
        mock_judge = AsyncMock()
        mock_judge.evaluate.side_effect = RuntimeError("LLM timeout")

        with patch("app.graph.nodes.llm_judge.LLMJudge", return_value=mock_judge):
            result = await llm_judge_node(_state())

        assert "judge_result" not in result
        assert len(result["errors"]) == 1
        assert "LLM judge evaluation failed" in result["errors"][0]["message"]

    @pytest.mark.asyncio
    async def test_passes_query_and_documents_to_judge(self):
        ev = _evaluation()
        mock_judge = AsyncMock()
        mock_judge.evaluate.return_value = (ev, compute_overall_score(ev))

        with patch("app.graph.nodes.llm_judge.LLMJudge", return_value=mock_judge):
            await llm_judge_node(_state())

        mock_judge.evaluate.assert_called_once()
        call_kwargs = mock_judge.evaluate.call_args.kwargs
        assert call_kwargs["query"] == "What is the answer?"
        assert call_kwargs["answer"] == "The answer is 42."


# ── TestAutoApprovalGateVeto ──────────────────────────────────────────────────


class TestAutoApprovalGateVeto:
    @pytest.mark.asyncio
    async def test_auto_approved_when_both_pass(self):
        # confidence ~0.855 (> 0.70), judge 0.85 (> 0.70) → auto-approved
        result = await auto_approval_gate_node(_state(judge_result=_judge_result(0.85)))
        assert result["approval_status"] == "approved"
        assert result["auto_approved"] is True

    @pytest.mark.asyncio
    async def test_not_approved_when_confidence_fails(self):
        # Low confidence scores → overall < 0.70
        state = _state(
            router_conf=0.3,
            retrieval_conf=0.3,
            answer_conf=0.3,
            judge_result=_judge_result(0.95),
        )
        result = await auto_approval_gate_node(state)
        assert result.get("approval_status") != "approved"
        assert result["auto_approved"] is False

    @pytest.mark.asyncio
    async def test_not_approved_when_judge_vetoes(self):
        # High confidence but judge score < 0.70 → judge vetoes
        state = _state(
            router_conf=0.95,
            retrieval_conf=0.95,
            answer_conf=0.95,
            judge_result=_judge_result(0.50, recommendation="needs_review"),
        )
        result = await auto_approval_gate_node(state)
        assert result.get("approval_status") != "approved"
        assert result["auto_approved"] is False

    @pytest.mark.asyncio
    async def test_fallback_confidence_only_when_judge_absent(self):
        # No judge_result — falls back to confidence-only (high confidence → auto-approve)
        state = _state(router_conf=0.9, retrieval_conf=0.9, answer_conf=0.9)
        result = await auto_approval_gate_node(state)
        assert result["approval_status"] == "approved"
        assert result["auto_approved"] is True

    @pytest.mark.asyncio
    async def test_approval_comment_includes_judge_score(self):
        result = await auto_approval_gate_node(_state(judge_result=_judge_result(0.85)))
        comment = result["approval_record"]["comment"]
        assert "judge" in comment.lower() or "85%" in comment

    @pytest.mark.asyncio
    async def test_approval_comment_omits_judge_when_absent(self):
        state = _state(router_conf=0.9, retrieval_conf=0.9, answer_conf=0.9)
        result = await auto_approval_gate_node(state)
        comment = result["approval_record"]["comment"]
        assert "judge" not in comment.lower()

    @pytest.mark.asyncio
    async def test_boundary_judge_exactly_at_threshold(self):
        # 0.70 exactly should pass
        state = _state(judge_result=_judge_result(0.70))
        result = await auto_approval_gate_node(state)
        assert result["approval_status"] == "approved"

    @pytest.mark.asyncio
    async def test_boundary_judge_just_below_threshold(self):
        # 0.699 should fail
        state = _state(judge_result=_judge_result(0.699))
        result = await auto_approval_gate_node(state)
        assert result.get("approval_status") != "approved"
