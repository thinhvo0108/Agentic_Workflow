"""
Reranker service — scores (query, document) pairs with BAAI/bge-reranker-large.

Architecture
------------
CrossEncoder.predict() is CPU/GPU-bound and synchronous.  It runs inside
asyncio.to_thread() so the event loop is never blocked.

Model lifecycle
---------------
Loading bge-reranker-large takes several seconds and allocates ~1 GB.
The model is therefore cached at the class level so it is loaded once per
process regardless of how many times RerankerService() is instantiated.

An injected cross_encoder argument bypasses class-level caching entirely,
which lets tests pass a lightweight mock without touching HuggingFace.

Score normalisation
-------------------
bge-reranker-large returns raw logits (unbounded floats).  We apply sigmoid
so every score lands in (0, 1) and is comparable to the cosine similarity
scores produced by the retrieval step.
"""

import asyncio
import math
from typing import Any

from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.exceptions import RerankingError
from app.core.logging import get_logger
from app.graph.state import RankedDocument, RetrievedDocument

_logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _to_ranked(doc: RetrievedDocument, rerank_score: float) -> RankedDocument:
    return RankedDocument(
        id=doc["id"],
        content=doc["content"],
        source=doc["source"],
        metadata=doc["metadata"],
        retrieval_score=doc["score"],
        rerank_score=rerank_score,
    )


# ── Service ───────────────────────────────────────────────────────────────────


class RerankerService:
    """Reranks a list of RetrievedDocuments using a CrossEncoder model.

    Parameters
    ----------
    cross_encoder:
        An object with a ``predict(pairs, **kwargs)`` method.  When None
        (the default in production), the configured model is loaded lazily
        and cached at the class level.  Pass a mock instance in tests.
    """

    # ── Class-level model cache ────────────────────────────────────────────────
    # Shared across every instance so the ~1 GB model is loaded only once.
    _cached_model: Any = None
    _cache_lock: asyncio.Lock | None = None

    def __init__(self, cross_encoder: Any | None = None) -> None:
        self._settings = get_settings()
        self._injected: Any | None = cross_encoder

    # ── Internal helpers ───────────────────────────────────────────────────────

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Return the class-level asyncio.Lock, creating it lazily."""
        if cls._cache_lock is None:
            cls._cache_lock = asyncio.Lock()
        return cls._cache_lock

    async def _get_model(self) -> Any:
        """Return the cross-encoder model, loading it on first call."""
        if self._injected is not None:
            return self._injected

        lock = self._get_lock()
        async with lock:
            if RerankerService._cached_model is None:
                model_name = self._settings.rag.reranker_model
                _logger.info("loading_reranker_model", model=model_name)
                try:
                    RerankerService._cached_model = await asyncio.to_thread(
                        CrossEncoder, model_name
                    )
                except Exception as exc:
                    raise RerankingError(
                        f"Failed to load reranker model '{model_name}': {exc}"
                    ) from exc
                _logger.info("reranker_model_loaded", model=model_name)

        return RerankerService._cached_model

    # ── Public API ─────────────────────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_n: int | None = None,
    ) -> list[RankedDocument]:
        """Score every document against *query* and return the top *top_n*.

        Parameters
        ----------
        query:
            The user's original search query.
        documents:
            Candidate documents from the retrieval step.
        top_n:
            Number of documents to return.  Defaults to
            ``settings.rag.reranker_top_n``.  Capped at ``len(documents)``
            so callers never receive an IndexError.

        Returns
        -------
        list[RankedDocument]
            Sorted descending by ``rerank_score``, length ≤ top_n.

        Raises
        ------
        RerankingError
            If the model cannot be loaded or ``predict`` raises.
        ValueError
            If *query* is blank.
        """
        if not query.strip():
            raise ValueError("query must not be blank")

        if not documents:
            return []

        n = top_n if top_n is not None else self._settings.rag.reranker_top_n
        n = min(n, len(documents))

        model = await self._get_model()

        pairs = [(query, doc["content"]) for doc in documents]

        try:
            raw_scores = await asyncio.to_thread(
                model.predict,
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError(f"CrossEncoder.predict failed: {exc}") from exc

        # raw_scores may be numpy array; convert element-wise to Python floats
        scored: list[tuple[RetrievedDocument, float]] = [
            (doc, _sigmoid(float(score))) for doc, score in zip(documents, raw_scores, strict=False)
        ]

        scored.sort(key=lambda t: t[1], reverse=True)

        result = [_to_ranked(doc, score) for doc, score in scored[:n]]

        _logger.info(
            "reranking_complete",
            input_count=len(documents),
            top_n=n,
            top_score=result[0]["rerank_score"] if result else None,
        )
        return result

    async def warm_up(self) -> None:
        """Load (or confirm) the reranker model is in the class-level cache.

        Call this during application startup so the first real request doesn't
        pay the 30-60 s load cost.
        """
        await self._get_model()

    # ── Test utility ──────────────────────────────────────────────────────────

    @classmethod
    def _reset_model_cache(cls) -> None:
        """Clear the class-level model cache.  Use only in tests."""
        cls._cached_model = None
        cls._cache_lock = None
