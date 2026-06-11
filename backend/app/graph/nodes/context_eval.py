"""Context precision evaluation node — RAGAS Context Precision pillar.

Runs after the reranker and before the generator. Asks the LLM whether each
retrieved document is relevant to the query.

    context_precision_score = relevant_docs / total_docs

The node is non-blocking: on failure the error is appended to state["errors"]
and the workflow continues with unaffected generation.
"""

from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.evaluation.context_precision import ContextPrecisionEvaluator
from app.graph.state import (
    AppState,
    ContextPrecisionResult,
    DocumentRelevanceVerdict,
    make_error,
)
from app.observability.token_tracker import TokenCounterCallback, instrumented_llm

_logger = get_logger(__name__)
_NODE = "context_eval"


async def context_eval_node(state: AppState) -> dict[str, Any]:
    """Evaluate whether retrieved documents are relevant to the query.

    Reads
    -----
    query, reranked_documents

    Writes
    ------
    context_precision  : ContextPrecisionResult (None on failure)
    current_node       : str
    step_count         : incremented by 1
    errors             : appended on failure
    total_tokens       : accumulated
    """
    _logger.info("context_eval_node_start", session_id=state["session_id"])

    step = state.get("step_count", 0) + 1
    documents = state.get("reranked_documents") or []

    if not documents:
        _logger.warning("context_eval_node_skip_no_docs", session_id=state["session_id"])
        return {"current_node": _NODE, "step_count": step}

    try:
        counter = TokenCounterCallback()
        evaluator = ContextPrecisionEvaluator(llm=instrumented_llm(counter))
        evaluation = await evaluator.evaluate(
            query=state["query"],
            documents=documents,
        )
    except Exception as exc:
        _logger.error(
            "context_eval_node_failed",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, f"Context precision evaluation failed: {exc}")],
        }

    relevant: list[DocumentRelevanceVerdict] = []
    irrelevant: list[DocumentRelevanceVerdict] = []

    for verdict in evaluation.verdicts:
        entry = DocumentRelevanceVerdict(
            document_id=verdict.document_id,
            is_relevant=verdict.is_relevant,
            reasoning=verdict.reasoning,
        )
        (relevant if verdict.is_relevant else irrelevant).append(entry)

    total = len(evaluation.verdicts)
    score = round(len(relevant) / total, 4) if total > 0 else 0.0

    result: ContextPrecisionResult = {
        "context_precision_score": score,
        "relevant_documents": relevant,
        "irrelevant_documents": irrelevant,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }

    _logger.info(
        "context_eval_node_done",
        session_id=state["session_id"],
        score=score,
        relevant=len(relevant),
        irrelevant=len(irrelevant),
    )
    return {
        "context_precision": result,
        "current_node": _NODE,
        "step_count": step,
        "total_tokens": counter.total,
    }
