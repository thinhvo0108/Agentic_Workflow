"""
Unit tests for app.services.confidence.

All functions are pure; no mocking required.

Test groups
-----------
TestClamp           — boundary clamping and 4-dp rounding
TestScoreRetrieval  — empty input, single doc, position-weighting behaviour
TestScoreAnswer     — empty input, single doc, mean computation
TestScoreOverall    — weight contract, edge cases, mid-value spot-check
"""

import pytest

from app.graph.state import RankedDocument, RetrievedDocument
from app.services.confidence import _clamp, score_answer, score_overall, score_retrieval

# ── Factories ──────────────────────────────────────────────────────────────────


def _rdoc(score: float, doc_id: str = "d1") -> RetrievedDocument:
    return RetrievedDocument(id=doc_id, content="text", source="src", metadata={}, score=score)


def _ranked(rerank_score: float, doc_id: str = "d1") -> RankedDocument:
    return RankedDocument(
        id=doc_id,
        content="text",
        source="src",
        metadata={},
        retrieval_score=0.5,
        rerank_score=rerank_score,
    )


# ── _clamp ─────────────────────────────────────────────────────────────────────


class TestClamp:
    def test_midpoint_unchanged(self):
        assert _clamp(0.5) == pytest.approx(0.5)

    def test_above_one_clamped_to_one(self):
        assert _clamp(1.5) == 1.0

    def test_below_zero_clamped_to_zero(self):
        assert _clamp(-0.3) == 0.0

    def test_zero_boundary(self):
        assert _clamp(0.0) == 0.0

    def test_one_boundary(self):
        assert _clamp(1.0) == 1.0

    def test_rounds_to_four_decimal_places(self):
        # 0.123456789 rounds to 0.1235
        result = _clamp(0.123456789)
        assert result == pytest.approx(0.1235, abs=1e-8)

    def test_fifth_decimal_is_dropped(self):
        result = _clamp(0.12344)
        assert result == pytest.approx(0.1234, abs=1e-8)


# ── score_retrieval ────────────────────────────────────────────────────────────


class TestScoreRetrieval:
    def test_empty_list_returns_zero(self):
        assert score_retrieval([]) == 0.0

    def test_single_perfect_score(self):
        assert score_retrieval([_rdoc(1.0)]) == pytest.approx(1.0)

    def test_single_zero_score(self):
        assert score_retrieval([_rdoc(0.0)]) == pytest.approx(0.0)

    def test_single_mid_score_passthrough(self):
        assert score_retrieval([_rdoc(0.6)]) == pytest.approx(0.6)

    def test_position_weighting_first_doc_dominates(self):
        # Rank-1 doc: score=1.0 (weight 1.0); rank-2 doc: score=0.0 (weight 0.5)
        # → weighted mean = (1.0*1.0 + 0.5*0.0) / 1.5 = 0.667
        high_first = score_retrieval([_rdoc(1.0, "d1"), _rdoc(0.0, "d2")])

        # Rank-1 doc: score=0.0 (weight 1.0); rank-2 doc: score=1.0 (weight 0.5)
        # → weighted mean = (1.0*0.0 + 0.5*1.0) / 1.5 = 0.333
        low_first = score_retrieval([_rdoc(0.0, "d1"), _rdoc(1.0, "d2")])

        assert high_first > low_first

    def test_two_equal_docs_return_their_score(self):
        # When both docs share the same score the weighted mean equals that score.
        docs = [_rdoc(0.8, "d1"), _rdoc(0.8, "d2")]
        assert score_retrieval(docs) == pytest.approx(0.8, abs=1e-3)

    def test_all_perfect_scores(self):
        docs = [_rdoc(1.0, f"d{i}") for i in range(5)]
        assert score_retrieval(docs) == pytest.approx(1.0)

    def test_all_zero_scores(self):
        docs = [_rdoc(0.0, f"d{i}") for i in range(5)]
        assert score_retrieval(docs) == pytest.approx(0.0)

    def test_result_always_in_unit_interval(self):
        docs = [_rdoc(0.9, "d1"), _rdoc(0.5, "d2"), _rdoc(0.2, "d3")]
        result = score_retrieval(docs)
        assert 0.0 <= result <= 1.0

    def test_three_docs_spot_check(self):
        # Weights: 1, 0.5, 0.333…  total = 1.833…
        # scores:  0.9, 0.6, 0.3
        # weighted sum = 1*0.9 + 0.5*0.6 + 0.333*0.3 = 0.9 + 0.3 + 0.1 = 1.3
        # mean = 1.3 / 1.833 ≈ 0.7091
        docs = [_rdoc(0.9, "d1"), _rdoc(0.6, "d2"), _rdoc(0.3, "d3")]
        weights = [1.0, 0.5, 1 / 3]
        expected = sum(w * s for w, s in zip(weights, [0.9, 0.6, 0.3], strict=False)) / sum(weights)
        assert score_retrieval(docs) == pytest.approx(expected, abs=1e-3)


# ── score_answer ──────────────────────────────────────────────────────────────


class TestScoreAnswer:
    def test_empty_list_returns_zero(self):
        assert score_answer([]) == 0.0

    def test_single_doc_passthrough(self):
        assert score_answer([_ranked(0.75)]) == pytest.approx(0.75)

    def test_perfect_score(self):
        assert score_answer([_ranked(1.0), _ranked(1.0)]) == pytest.approx(1.0)

    def test_zero_score(self):
        assert score_answer([_ranked(0.0), _ranked(0.0)]) == pytest.approx(0.0)

    def test_max_of_three(self):
        docs = [_ranked(0.8, "d1"), _ranked(0.6, "d2"), _ranked(0.4, "d3")]
        assert score_answer(docs) == pytest.approx(0.8, abs=1e-3)

    def test_result_always_in_unit_interval(self):
        docs = [_ranked(0.9), _ranked(0.7), _ranked(0.55)]
        result = score_answer(docs)
        assert 0.0 <= result <= 1.0

    def test_order_does_not_affect_result(self):
        # max() is order-independent, same as mean was.
        docs_asc = [_ranked(0.3, "d1"), _ranked(0.6, "d2"), _ranked(0.9, "d3")]
        docs_desc = [_ranked(0.9, "d1"), _ranked(0.6, "d2"), _ranked(0.3, "d3")]
        assert score_answer(docs_asc) == pytest.approx(score_answer(docs_desc))


# ── score_overall ─────────────────────────────────────────────────────────────


class TestScoreOverall:
    def test_all_perfect_is_one(self):
        assert score_overall(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_all_zero_is_zero(self):
        assert score_overall(0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_uniform_input_returns_same_value(self):
        # Weights sum to 1.0, so a uniform input v should return v.
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert score_overall(v, v, v) == pytest.approx(v, abs=1e-4)

    def test_answer_has_most_weight(self):
        # Isolate each signal to confirm answer (0.50) > retrieval (0.30) > router (0.20)
        answer_only = score_overall(router=0.0, retrieval=0.0, answer=1.0)
        retrieval_only = score_overall(router=0.0, retrieval=1.0, answer=0.0)
        router_only = score_overall(router=1.0, retrieval=0.0, answer=0.0)
        assert answer_only > retrieval_only > router_only

    def test_answer_weight_is_0_50(self):
        result = score_overall(router=0.0, retrieval=0.0, answer=1.0)
        assert result == pytest.approx(0.50, abs=1e-4)

    def test_retrieval_weight_is_0_30(self):
        result = score_overall(router=0.0, retrieval=1.0, answer=0.0)
        assert result == pytest.approx(0.30, abs=1e-4)

    def test_router_weight_is_0_20(self):
        result = score_overall(router=1.0, retrieval=0.0, answer=0.0)
        assert result == pytest.approx(0.20, abs=1e-4)

    def test_mid_values_spot_check(self):
        # 0.2*0.8 + 0.3*0.6 + 0.5*0.9 = 0.16 + 0.18 + 0.45 = 0.79
        result = score_overall(router=0.8, retrieval=0.6, answer=0.9)
        assert result == pytest.approx(0.79, abs=1e-3)

    def test_result_always_in_unit_interval(self):
        for r, ret, ans in [(0.5, 0.7, 0.6), (0.1, 0.9, 0.3), (1.0, 0.0, 0.5)]:
            result = score_overall(r, ret, ans)
            assert 0.0 <= result <= 1.0
