"""LLM-as-a-judge node.

Runs after groundedness and before checkpoint. Uses a separate LLM call to
evaluate the generated answer on four quality dimensions:

  faithfulness  — claims are supported by source documents (weight 0.40)
  relevance     — answer directly addresses the query       (weight 0.30)
  completeness  — all key aspects are covered               (weight 0.20)
  coherence     — well-structured and internally consistent (weight 0.10)

The overall_score feeds into auto_approval_gate as a veto signal:
  auto-approve = confidence >= threshold AND judge_score >= threshold

The node is non-blocking: if the LLM call fails the error is appended to
state["errors"] and the workflow continues. auto_approval_gate falls back to
confidence-only when judge_result is None (graceful degradation).
"""

from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.evaluation.judge import LLMJudge
from app.graph.state import AppState, JudgeDimensionScore, JudgeResult, make_error
from app.observability.token_tracker import TokenCounterCallback, instrumented_llm

_logger = get_logger(__name__)
_NODE = "llm_judge"

_AUTO_APPROVE_THRESHOLD = 0.70


async def llm_judge_node(state: AppState) -> dict[str, Any]:
    """Evaluate the generated answer quality using an LLM judge.

    Reads
    -----
    query, structured_output, reranked_documents

    Writes
    ------
    judge_result  : JudgeResult (None on failure — workflow continues)
    current_node  : str
    step_count    : incremented by 1
    errors        : appended on failure
    """
    _logger.info("llm_judge_node_start", session_id=state["session_id"])

    step = state.get("step_count", 0) + 1
    so = state.get("structured_output")

    if so is None:
        _logger.warning("llm_judge_node_skip_no_output", session_id=state["session_id"])
        return {"current_node": _NODE, "step_count": step}

    answer = so.get("answer", "")
    if not answer:
        _logger.warning("llm_judge_node_skip_empty_answer", session_id=state["session_id"])
        return {"current_node": _NODE, "step_count": step}

    documents = state.get("reranked_documents") or []

    try:
        counter = TokenCounterCallback()
        judge = LLMJudge(llm=instrumented_llm(counter))
        evaluation, overall_score = await judge.evaluate(
            query=state["query"],
            answer=answer,
            documents=documents,
        )
    except Exception as exc:
        _logger.error(
            "llm_judge_node_failed",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, f"LLM judge evaluation failed: {exc}")],
        }

    result: JudgeResult = {
        "faithfulness": JudgeDimensionScore(
            score=evaluation.faithfulness.score,
            reasoning=evaluation.faithfulness.reasoning,
        ),
        "relevance": JudgeDimensionScore(
            score=evaluation.relevance.score,
            reasoning=evaluation.relevance.reasoning,
        ),
        "completeness": JudgeDimensionScore(
            score=evaluation.completeness.score,
            reasoning=evaluation.completeness.reasoning,
        ),
        "coherence": JudgeDimensionScore(
            score=evaluation.coherence.score,
            reasoning=evaluation.coherence.reasoning,
        ),
        "overall_score": overall_score,
        "recommendation": "auto_approve"
        if overall_score >= _AUTO_APPROVE_THRESHOLD
        else "needs_review",
        "critique": evaluation.critique,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }

    _logger.info(
        "llm_judge_node_done",
        session_id=state["session_id"],
        overall_score=overall_score,
        recommendation=result["recommendation"],
        faithfulness=evaluation.faithfulness.score,
        relevance=evaluation.relevance.score,
        completeness=evaluation.completeness.score,
        coherence=evaluation.coherence.score,
    )
    return {
        "judge_result": result,
        "current_node": _NODE,
        "step_count": step,
        "total_tokens": counter.total,
    }
