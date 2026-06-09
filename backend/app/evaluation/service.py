"""Groundedness evaluation service — bridges the evaluator and graph state."""

from datetime import UTC, datetime

from app.core.logging import get_logger
from app.evaluation.evaluator import GroundednessEvaluator
from app.evaluation.schemas import GroundednessEvaluation
from app.graph.state import EvaluatedClaim, GroundednessResult, RankedDocument

_logger = get_logger(__name__)


def build_groundedness_result(evaluation: GroundednessEvaluation) -> GroundednessResult:
    """Convert a Pydantic evaluation into a plain-dict GroundednessResult.

    Score computation
    -----------------
    groundedness_score = len(supported_claims) / total_claims

    Using count-based scoring rather than asking the LLM for a float directly
    avoids numeric inconsistency: the hard classification is done by the model
    on a binary scale; the numeric aggregation is deterministic on our side.
    """
    supported: list[EvaluatedClaim] = []
    unsupported: list[EvaluatedClaim] = []

    for verdict in evaluation.claims:
        entry = EvaluatedClaim(
            claim=verdict.claim,
            supported=verdict.supported,
            source_document_ids=verdict.source_document_ids,
            reasoning=verdict.reasoning,
        )
        (supported if verdict.supported else unsupported).append(entry)

    total = len(evaluation.claims)
    score = round(len(supported) / total, 4) if total > 0 else 0.0

    return GroundednessResult(
        groundedness_score=score,
        supported_claims=supported,
        unsupported_claims=unsupported,
        evaluated_at=datetime.now(UTC).isoformat(),
    )


async def evaluate_groundedness(
    query: str,
    answer: str,
    documents: list[RankedDocument],
    evaluator: GroundednessEvaluator | None = None,
) -> GroundednessResult:
    """Run groundedness evaluation and return a serialisable result dict.

    Parameters
    ----------
    query:
        Original user question.
    answer:
        Generated answer text.
    documents:
        Reranked source documents.
    evaluator:
        Injected evaluator instance (tests pass a mock).

    Returns
    -------
    GroundednessResult
        Plain TypedDict — safe to store in AppState and checkpoint.
    """
    ev = evaluator or GroundednessEvaluator()
    evaluation = await ev.evaluate(query=query, answer=answer, documents=documents)
    result = build_groundedness_result(evaluation)
    _logger.info(
        "groundedness_service_done",
        score=result["groundedness_score"],
        supported=len(result["supported_claims"]),
        unsupported=len(result["unsupported_claims"]),
    )
    return result
