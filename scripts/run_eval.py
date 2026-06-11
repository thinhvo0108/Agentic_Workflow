#!/usr/bin/env python3
"""
Offline evaluation harness for the Agentic Workflow.

Usage
-----
    # The stack must be running first (docker compose up)
    cd /path/to/project/backend
    uv run python ../scripts/run_eval.py

    # Or with options:
    uv run python ../scripts/run_eval.py \\
        --base-url   http://localhost:8000 \\
        --ollama-url http://localhost:11434 \\
        --dataset    ../data/eval_dataset.yaml \\
        --output     ../data/eval_results \\
        --concurrency 2 \\
        --timeout    300

    # Skip RAGAS (faster, no ragas package needed):
    uv run python ../scripts/run_eval.py --no-ragas

What it does
------------
1. Loads questions + ground-truth from the YAML dataset.
2. Submits each query to POST /api/v1/workflow.
3. Polls status until completed or awaiting_approval.
4. Auto-approves awaiting_approval cases (reviewer_id: "eval-harness") so
   the full pipeline runs end-to-end without manual intervention.
5. Fetches the final WorkflowResponse for each completed case.
6. Computes answer similarity vs ground_truth via Ollama embeddings (cosine).
7. Optionally runs RAGAS scoring if `ragas` + `datasets` are installed.
8. Writes a timestamped JSON results file to --output.
9. Prints a per-case and aggregate markdown report.

Metrics collected per case
--------------------------
  auto_approved           bool     — did the pipeline auto-approve without human?
  judge_score             float    — LLM-as-a-judge overall score
  groundedness_score      float    — fraction of answer claims grounded in sources
  context_precision_score float    — fraction of retrieved docs relevant to query
  hallucination_rate      float    — 1 - groundedness_score
  answer_similarity       float    — cosine sim of answer vs ground_truth (Ollama)
  routing_correct         bool     — did the router pick the expected agent_type?
  latency_ms              float    — end-to-end wall-clock time
  total_tokens            int      — prompt + completion tokens across all nodes

Aggregate metrics
-----------------
  auto_approval_rate       % of cases auto-approved without human review
  routing_accuracy         % of cases routed to expected agent
  mean_judge_score         average LLM judge score
  mean_groundedness        average groundedness score
  mean_context_precision   average context precision score
  mean_answer_similarity   average semantic similarity to ground truth
  mean_hallucination_rate  average hallucination rate
  mean_latency_ms          average end-to-end latency
  mean_total_tokens        average token usage
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ── Optional: add backend to sys.path so ragas_scorer can be imported ─────────
_BACKEND = Path(__file__).parent.parent / "backend"
if _BACKEND.exists() and str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

try:
    import httpx
except ImportError:
    print("httpx is required: uv add httpx (or pip install httpx)", file=sys.stderr)
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("pyyaml is required: uv add pyyaml (or pip install pyyaml)", file=sys.stderr)
    sys.exit(1)


# ── Configuration ──────────────────────────────────────────────────────────────

_SCRIPT_DIR   = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_BASE_URL   = "http://localhost:8000"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_DATASET    = _PROJECT_ROOT / "data" / "eval_dataset.yaml"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "eval_results"

POLL_INTERVAL  = 2.0    # seconds between status polls
EVAL_REVIEWER  = "eval-harness"


# ── Ollama embedding + cosine similarity ──────────────────────────────────────

async def _embed(client: httpx.AsyncClient, ollama_url: str, text: str) -> list[float] | None:
    """Embed text via Ollama. Tries /api/embed (v0.3+) then /api/embeddings."""
    for path, body, key in [
        ("/api/embed",       {"model": "nomic-embed-text", "input":  text}, "embeddings"),
        ("/api/embeddings",  {"model": "nomic-embed-text", "prompt": text}, "embedding"),
    ]:
        try:
            r = await client.post(f"{ollama_url}{path}", json=body, timeout=30.0)
            if r.status_code == 200:
                raw = r.json()[key]
                return raw[0] if isinstance(raw[0], list) else raw
        except Exception:
            continue
    return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    na    = math.sqrt(sum(x * x for x in a))
    nb    = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── API helpers ────────────────────────────────────────────────────────────────

async def _submit(client: httpx.AsyncClient, base_url: str, query: str) -> str:
    r = await client.post(f"{base_url}/api/v1/workflow", json={"query": query}, timeout=30.0)
    r.raise_for_status()
    return r.json()["session_id"]


async def _poll(
    client: httpx.AsyncClient,
    base_url: str,
    session_id: str,
    timeout: float,
) -> str:
    deadline = time.monotonic() + timeout
    while True:
        r = await client.get(f"{base_url}/api/v1/workflow/{session_id}", timeout=10.0)
        r.raise_for_status()
        status = r.json()["status"]
        if status != "running":
            return status
        if time.monotonic() >= deadline:
            return "timeout"
        await asyncio.sleep(POLL_INTERVAL)


async def _approve(client: httpx.AsyncClient, base_url: str, session_id: str) -> None:
    r = await client.post(
        f"{base_url}/api/v1/workflow/{session_id}/approve",
        json={"session_id": session_id, "action": "approved", "reviewer_id": EVAL_REVIEWER},
        timeout=180.0,
    )
    r.raise_for_status()


async def _result(client: httpx.AsyncClient, base_url: str, session_id: str) -> dict[str, Any]:
    r = await client.get(f"{base_url}/api/v1/workflow/{session_id}/result", timeout=30.0)
    r.raise_for_status()
    return r.json()


# ── Run one eval case ──────────────────────────────────────────────────────────

async def _run_case(
    case: dict[str, Any],
    base_url: str,
    ollama_url: str,
    timeout: float,
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    cid      = case.get("id", "?")
    question = case["question"]
    gt       = case.get("ground_truth", "")
    expected = case.get("agent_type")

    out: dict[str, Any] = {
        "id": cid, "question": question[:80], "expected_agent": expected,
        "actual_agent": None, "status": "error",
        "auto_approved": None, "judge_score": None,
        "groundedness_score": None, "context_precision_score": None,
        "hallucination_rate": None, "answer_similarity": None,
        "routing_correct": None, "latency_ms": None, "total_tokens": None,
        "answer_excerpt": None, "error": None,
    }

    async with sem:
        try:
            sid    = await _submit(client, base_url, question)
            status = await _poll(client, base_url, sid, timeout)

            if status == "awaiting_approval":
                print(f"  [{cid}] awaiting_approval — auto-approving...")
                await _approve(client, base_url, sid)
                status = await _poll(client, base_url, sid, 180.0)

            out["status"] = status

            if status in ("completed", "approved"):
                data = _result.__func__ if False else None  # just a hint
                data = await _result(client, base_url, sid)

                metrics = data.get("metrics") or {}
                gnd     = data.get("groundedness") or {}
                cp      = data.get("context_precision") or {}
                answer  = data.get("answer") or ""

                out.update({
                    "actual_agent":             data.get("route"),
                    "auto_approved":            data.get("auto_approved"),
                    "judge_score":              metrics.get("judge_score"),
                    "groundedness_score":       gnd.get("groundedness_score"),
                    "context_precision_score":  cp.get("context_precision_score"),
                    "hallucination_rate":       metrics.get("hallucination_rate"),
                    "latency_ms":               metrics.get("latency_ms"),
                    "total_tokens":             metrics.get("total_tokens"),
                    "answer_excerpt":           answer[:120],
                })

                if expected and out["actual_agent"]:
                    out["routing_correct"] = (out["actual_agent"] == expected)

                # Answer similarity via Ollama embeddings
                if gt and answer:
                    ea = await _embed(client, ollama_url, answer)
                    eg = await _embed(client, ollama_url, gt)
                    if ea and eg:
                        out["answer_similarity"] = round(_cosine(ea, eg), 4)

        except Exception as exc:
            out["error"] = str(exc)

        label = "✓" if out.get("auto_approved") else "⚑" if out["status"] == "awaiting_approval" else out["status"]
        print(f"  [{cid}] {label}  judge={_pct(out.get('judge_score'))}  "
              f"grounded={_pct(out.get('groundedness_score'))}  "
              f"sim={_pct(out.get('answer_similarity'))}  "
              f"lat={_ms(out.get('latency_ms'))}")

    return out


# ── Aggregate ─────────────────────────────────────────────────────────────────

def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in results if r["status"] in ("completed", "approved")]

    def mean(vals: list[Any]) -> float | None:
        vs = [v for v in vals if v is not None]
        return round(sum(vs) / len(vs), 4) if vs else None

    auto_count = sum(1 for r in done if r.get("auto_approved"))
    routing = [r["routing_correct"] for r in done if r.get("routing_correct") is not None]

    return {
        "total_cases":             len(results),
        "completed":               len(done),
        "auto_approved_count":     auto_count,
        "auto_approval_rate":      round(auto_count / len(done), 4) if done else None,
        "routing_accuracy":        mean([float(v) for v in routing]) if routing else None,
        "mean_judge_score":        mean([r.get("judge_score")             for r in done]),
        "mean_groundedness":       mean([r.get("groundedness_score")      for r in done]),
        "mean_context_precision":  mean([r.get("context_precision_score") for r in done]),
        "mean_hallucination_rate": mean([r.get("hallucination_rate")      for r in done]),
        "mean_answer_similarity":  mean([r.get("answer_similarity")       for r in done]),
        "mean_latency_ms":         mean([r.get("latency_ms")              for r in done]),
        "mean_total_tokens":       mean([r.get("total_tokens")            for r in done]),
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    return f"{round(v * 100):>3}%" if v is not None else "  —"

def _ms(v: float | None) -> str:
    return f"{v / 1000:.1f}s" if v is not None else "—"

def _tok(v: float | None) -> str:
    return f"{int(v):,}" if v is not None else "—"


# ── Markdown report ────────────────────────────────────────────────────────────

def _print_report(results: list[dict[str, Any]], agg: dict[str, Any], elapsed: float) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'=' * 76}")
    print(f"  AGENTIC WORKFLOW — EVAL REPORT   {ts}")
    print(f"{'=' * 76}\n")

    hdrs  = ["ID", "Agent", "Status", "Auto", "Judge", "Grnd", "CtxP", "Sim", "Lat",  "Tok"]
    widths = [5,   8,       10,       5,      7,       6,      6,      6,     7,      7]
    sep    = "  "
    header = sep.join(h.ljust(w) for h, w in zip(hdrs, widths))
    print(header)
    print("─" * len(header))

    for r in results:
        auto = ("✓" if r.get("auto_approved") else "✗") if r.get("auto_approved") is not None else "—"
        row = [
            str(r["id"]).ljust(widths[0]),
            (r.get("expected_agent") or "?")[:widths[1]].ljust(widths[1]),
            r["status"][:widths[2]].ljust(widths[2]),
            auto.ljust(widths[3]),
            _pct(r.get("judge_score")).ljust(widths[4]),
            _pct(r.get("groundedness_score")).ljust(widths[5]),
            _pct(r.get("context_precision_score")).ljust(widths[6]),
            _pct(r.get("answer_similarity")).ljust(widths[7]),
            _ms(r.get("latency_ms")).ljust(widths[8]),
            _tok(r.get("total_tokens")).ljust(widths[9]),
        ]
        print(sep.join(row))

    print(f"{'─' * len(header)}")
    print(f"\n  {agg['completed']}/{agg['total_cases']} cases completed   {elapsed:.1f}s total\n")

    rows = [
        ("Auto-approval rate",    _pct(agg.get("auto_approval_rate")),
         f"({agg['auto_approved_count']}/{agg['completed']} cases)"),
        ("Routing accuracy",      _pct(agg.get("routing_accuracy")),          ""),
        ("Mean judge score",      _pct(agg.get("mean_judge_score")),          "LLM holistic quality"),
        ("Mean groundedness",     _pct(agg.get("mean_groundedness")),         "answer claim support"),
        ("Mean ctx precision",    _pct(agg.get("mean_context_precision")),    "retrieval relevance"),
        ("Mean answer similarity",_pct(agg.get("mean_answer_similarity")),    "vs ground truth"),
        ("Mean hallucination",    _pct(agg.get("mean_hallucination_rate")),   "lower is better"),
        ("Mean latency",          _ms(agg.get("mean_latency_ms")),            ""),
        ("Mean tokens",           _tok(agg.get("mean_total_tokens")),         ""),
    ]
    for label, value, note in rows:
        note_str = f"  ({note})" if note else ""
        print(f"  {label:<26}  {value}{note_str}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

async def _main() -> int:
    ap = argparse.ArgumentParser(description="Agentic Workflow offline eval harness")
    ap.add_argument("--base-url",    default=DEFAULT_BASE_URL,   help="API base URL")
    ap.add_argument("--ollama-url",  default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    ap.add_argument("--dataset",     type=Path, default=DEFAULT_DATASET,    help="YAML dataset path")
    ap.add_argument("--output",      type=Path, default=DEFAULT_OUTPUT_DIR, help="Results dir")
    ap.add_argument("--concurrency", type=int,   default=2,    help="Parallel workflow cases")
    ap.add_argument("--timeout",     type=float, default=300.0, help="Seconds per case")
    ap.add_argument("--no-ragas",    action="store_true",       help="Skip RAGAS scoring")
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    with open(args.dataset) as f:
        raw = yaml.safe_load(f)
    cases: list[dict[str, Any]] = raw.get("cases", [])
    if not cases:
        print("No cases in dataset.", file=sys.stderr)
        return 1

    print(f"\nLoaded {len(cases)} eval cases from {args.dataset.name}")
    print(f"API: {args.base_url}  |  Ollama: {args.ollama_url}  |  Concurrency: {args.concurrency}\n")

    sem = asyncio.Semaphore(args.concurrency)
    t0  = time.monotonic()

    async with httpx.AsyncClient() as client:
        tasks = [
            _run_case(case, args.base_url, args.ollama_url, args.timeout, sem, client)
            for case in cases
        ]
        results: list[dict[str, Any]] = await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0
    agg = _aggregate(results)
    _print_report(results, agg, elapsed)

    # Optional RAGAS scoring
    if not args.no_ragas:
        try:
            from app.evaluation.ragas_scorer import run_ragas  # type: ignore[import]
            print("  Running RAGAS evaluation...")
            ragas_scores = await run_ragas(
                cases=cases,
                results=results,
                ollama_base_url=args.ollama_url,
            )
            if ragas_scores:
                print("\n  RAGAS SCORES\n  " + "─" * 38)
                for k, v in ragas_scores.items():
                    print(f"  {k:<30}  {_pct(v)}")
                print()
                agg["ragas"] = ragas_scores
        except ImportError:
            print(
                "  (ragas / datasets not installed — skipping RAGAS metrics)\n"
                "  Install: uv add ragas datasets --optional eval"
            )
        except Exception as exc:
            print(f"  (RAGAS scoring error: {exc})")

    # Save JSON results
    args.output.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_file = args.output / f"eval_{ts}.json"
    with open(out_file, "w") as f:
        json.dump({"cases": results, "aggregate": agg, "elapsed_s": round(elapsed, 2)}, f, indent=2)
    print(f"Results saved → {out_file}\n")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
