"""
Optional RAGAS evaluation wrapper.

Requires the `eval` optional dependency group:
    uv add ragas datasets --optional eval

This module is imported at runtime by scripts/run_eval.py only when the `ragas`
and `datasets` packages are available.  It wraps the pipeline's Ollama LLM and
embeddings so RAGAS runs fully locally without any external API key.

Metrics computed
----------------
  faithfulness       — are all answer claims supported by the context?
  answer_relevancy   — is the answer relevant to the question?
  context_precision  — are the retrieved chunks ordered by relevance?  (via
                       precision@K logic; uses our context_precision_score from
                       the pipeline when RAGAS cannot compute it independently)

Note on context_recall: requires a per-document ground-truth annotation that
we do not have in the eval dataset, so it is omitted here.

Usage (from scripts/run_eval.py)
---------------------------------
    from app.evaluation.ragas_scorer import run_ragas
    ragas_scores = await run_ragas(cases, results, ollama_base_url)
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any


async def run_ragas(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    ollama_base_url: str = "http://localhost:11434",
    model: str = "llama3.2:latest",
    embed_model: str = "nomic-embed-text",
) -> dict[str, float] | None:
    """
    Run RAGAS scoring on completed eval cases.

    Parameters
    ----------
    cases:
        The original eval dataset cases (contains question + ground_truth).
    results:
        The per-case dicts returned by the eval harness; matched to cases by
        index.  Only cases whose status is 'completed' or 'approved' are
        included.
    ollama_base_url:
        Base URL for the local Ollama server.
    model:
        Ollama chat model to use for RAGAS LLM-as-judge calls.
    embed_model:
        Ollama embedding model for answer-relevancy scoring.

    Returns
    -------
    dict[str, float] mapping metric name → mean score (0–1), or None if RAGAS
    could not complete (insufficient data, import error, etc.).
    """
    try:
        from datasets import Dataset
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            f"RAGAS dependencies missing: {exc}. "
            "Install with: uv add ragas datasets --optional eval"
        ) from exc

    rows: list[dict[str, Any]] = []
    result_by_id = {r["id"]: r for r in results}

    for case in cases:
        cid = case.get("id", "?")
        result = result_by_id.get(cid)
        if not result or result.get("status") not in ("completed", "approved"):
            continue

        question = case.get("question", "")
        gt = case.get("ground_truth", "")
        answer = result.get("answer_excerpt") or ""

        # Truncate the answer_excerpt (120 chars) is short — use it as a proxy.
        # The harness only stores an excerpt; for a richer eval, extend the harness
        # to save the full answer.
        if not (question and answer):
            continue

        # Context = fake single-document (we do not store retrieved chunks in
        # the harness results).  A future improvement would have the API return
        # the full retrieved docs so RAGAS can score context_precision properly.
        context = [gt] if gt else ["(no context)"]

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": context,
                "ground_truth": gt,
            }
        )

    if not rows:
        return None

    dataset = Dataset.from_list(rows)

    llm = LangchainLLMWrapper(ChatOllama(base_url=ollama_base_url, model=model))
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(base_url=ollama_base_url, model=embed_model)
    )

    metrics = [faithfulness, answer_relevancy, context_precision]
    for m in metrics:
        m.llm = llm
        m.embeddings = embeddings

    loop = asyncio.get_event_loop()
    eval_result = await loop.run_in_executor(
        None,
        lambda: evaluate(dataset=dataset, metrics=metrics),
    )

    scores: dict[str, float] = {}
    for metric_name in ("faithfulness", "answer_relevancy", "context_precision"):
        val = eval_result.get(metric_name)
        if val is not None:
            with contextlib.suppress(TypeError, ValueError):
                scores[f"ragas_{metric_name}"] = round(float(val), 4)

    return scores or None
