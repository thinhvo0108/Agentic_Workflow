"""
Confidence scoring utilities for the agentic workflow.

All functions are pure, stateless transformations over LangGraph state
types — no LLM calls, no I/O.  Each returns a float in [0.0, 1.0].

Scoring model
-------------
router_confidence
    The LLM's own estimate of how certain it is about the routing decision.
    Taken directly from RouteOutput.confidence.

retrieval_confidence
    Position-weighted mean of ChromaDB cosine similarity scores.  Documents
    are assumed to be ordered by descending similarity (rank 1 = most similar).
    Position weights follow a harmonic series (1, 1/2, 1/3 …) so the
    top-ranked document has the greatest influence.

answer_confidence
    Highest CrossEncoder rerank score among the documents fed to the generator.
    Using the maximum reflects that one highly-relevant document is sufficient
    for a strong answer — the mean unfairly penalises having additional context.

overall_confidence
    Weighted linear combination:
        router × 0.20 + retrieval × 0.30 + answer × 0.50
    Answer grounding carries the highest weight because it most directly
    measures whether the output is substantiated by evidence.
"""

from app.graph.state import RankedDocument, RetrievedDocument

_OVERALL_WEIGHTS: dict[str, float] = {
    "router": 0.20,
    "retrieval": 0.30,
    "answer": 0.50,
}


def score_retrieval(docs: list[RetrievedDocument]) -> float:
    """Position-weighted mean of ChromaDB cosine-similarity scores.

    Parameters
    ----------
    docs:
        Documents in descending similarity order (index 0 = best match).

    Returns
    -------
    float
        Score in [0.0, 1.0].  Returns 0.0 for an empty list.
    """
    if not docs:
        return 0.0
    weights = [1.0 / (i + 1) for i in range(len(docs))]
    total_weight = sum(weights)
    weighted_sum = sum(w * d["score"] for w, d in zip(weights, docs, strict=False))
    return _clamp(weighted_sum / total_weight)


def score_answer(docs: list[RankedDocument]) -> float:
    """Highest CrossEncoder rerank score among the context documents.

    Using the maximum rather than the mean reflects that the answer quality
    is bounded by the best available evidence — one highly-relevant document
    is sufficient for a strong answer regardless of other lower-scored docs.

    Parameters
    ----------
    docs:
        Reranked documents (in any order — only their scores matter).

    Returns
    -------
    float
        Score in [0.0, 1.0].  Returns 0.0 for an empty list.
    """
    if not docs:
        return 0.0
    return _clamp(max(d["rerank_score"] for d in docs))


def score_overall(router: float, retrieval: float, answer: float) -> float:
    """Weighted linear combination of the three confidence signals.

    Parameters
    ----------
    router, retrieval, answer:
        Individual confidence scores, each in [0.0, 1.0].

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    raw = (
        _OVERALL_WEIGHTS["router"] * router
        + _OVERALL_WEIGHTS["retrieval"] * retrieval
        + _OVERALL_WEIGHTS["answer"] * answer
    )
    return _clamp(raw)


def _clamp(value: float) -> float:
    """Round to 4 decimal places and clamp to [0.0, 1.0]."""
    return round(min(1.0, max(0.0, value)), 4)
