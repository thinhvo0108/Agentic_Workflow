"""
Workflow visualization utilities.

Generates human-readable representations of the compiled LangGraph state
machine for documentation, debugging, and portfolio presentation.

Usage
-----
    python -m app.graph.visualization          # prints all formats to stdout
    python -m app.graph.visualization --save   # writes docs/workflow_diagram.md
"""

from __future__ import annotations

import textwrap
from typing import Any


# ── Graph inspection ──────────────────────────────────────────────────────────


def _get_compiled_graph() -> Any:
    from langgraph.checkpoint.memory import MemorySaver
    from app.graph.workflow import compile_workflow
    return compile_workflow(MemorySaver())


def get_node_list() -> list[str]:
    """Return the ordered list of graph node names."""
    wf = _get_compiled_graph()
    return [n for n in wf.get_graph().nodes if not n.startswith("__")]


def get_edge_summary() -> list[dict[str, str]]:
    """Return a list of {source, target, type} dicts for every edge."""
    wf = _get_compiled_graph()
    edges = []
    for e in wf.get_graph().edges:
        edges.append({
            "source": e.source.lstrip("_"),
            "target": e.target.lstrip("_"),
            "type": "conditional" if e.conditional else "direct",
        })
    return edges


# ── Diagram generators ────────────────────────────────────────────────────────


def generate_mermaid() -> str:
    """Return the raw Mermaid flowchart string from LangGraph."""
    wf = _get_compiled_graph()
    return wf.get_graph().draw_mermaid()


def generate_ascii_table() -> str:
    """Return a text table describing every node and its role."""
    rows = [
        ("START",            "—",             "Entry point for every workflow run"),
        ("router",           "LLM",           "Classifies query → research | support"),
        ("research",         "pass-through",  "Marks the research execution path"),
        ("support",          "pass-through",  "Marks the support execution path"),
        ("retriever",        "ChromaDB",      "Fetches top-10 chunks by semantic similarity"),
        ("reranker",         "CrossEncoder",  "Re-scores chunks with BAAI/bge-reranker-large, keeps top-3"),
        ("generator",        "LLM",           "Produces summary + answer + citations (ResearchAgent or SupportAgent)"),
        ("structured_output","Pydantic",       "Validates draft JSON → StructuredOutput TypedDict"),
        ("checkpoint",       "PostgreSQL",    "Writes audit record to workflow_checkpoints table"),
        ("human_approval",   "interrupt ⏸",  "Graph pauses here; resumes after reviewer decision"),
        ("final_response",   "pure",          "Assembles FinalResponse TypedDict from approved output"),
        ("END",              "—",             "Workflow terminal state"),
    ]
    col_w = [max(len(r[i]) for r in rows) for i in range(3)]
    col_w = [max(cw, len(h)) for cw, h in zip(col_w, ["Node", "Backend", "Responsibility"])]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(["Node", "Backend", "Responsibility"], col_w)) + " |"
    lines = [sep, header, sep]
    for row in rows:
        lines.append("| " + " | ".join(v.ljust(w) for v, w in zip(row, col_w)) + " |")
    lines.append(sep)
    return "\n".join(lines)


def generate_flow_narrative() -> str:
    """Return a plain-text step-by-step description of the two execution paths."""
    return textwrap.dedent("""\
    Research path
    ─────────────
    1.  User submits a query via POST /api/v1/workflow.
    2.  router       → LLM classifies intent as "research".
    3.  research     → path is marked; query may be enriched.
    4.  retriever    → top-10 chunks fetched from ChromaDB.
    5.  reranker     → CrossEncoder re-scores, top-3 kept.
    6.  generator    → ResearchAgent calls Ollama; produces structured JSON.
    7.  structured_output → JSON validated against Pydantic schema.
    8.  checkpoint   → audit record written to PostgreSQL.
    9.  human_approval → graph PAUSES (interrupt_before).
    10. Reviewer calls POST /api/v1/workflow/{id}/approve.
    11. human_approval (resumed) → logs decision.
    12. final_response → FinalResponse assembled.
    13. Client calls GET /api/v1/workflow/{id}/result.

    Support path
    ────────────
    Steps 1–5 identical.
    6.  generator    → SupportAgent performs confidence triage:
                       • high confidence: answers directly (no citations)
                       • low confidence:  uses retrieved documents
    Steps 7–13 identical.

    Error / rejection paths
    ───────────────────────
    •  router failure  → errors list populated → conditional edge routes to END.
    •  reviewer rejects → approval_status = "rejected" → conditional edge routes to END.
    """)


def generate_full_document() -> str:
    """Compose the complete Markdown workflow documentation."""
    return "\n".join([
        "# Agentic Workflow — Graph Documentation",
        "",
        "Auto-generated by `app/graph/visualization.py`.",
        "Re-run `python -m app.graph.visualization --save` after any graph change.",
        "",
        "---",
        "",
        "## Mermaid Diagram",
        "",
        "```mermaid",
        generate_mermaid(),
        "```",
        "",
        "---",
        "",
        "## Node Reference",
        "",
        generate_ascii_table(),
        "",
        "---",
        "",
        "## Execution Paths",
        "",
        generate_flow_narrative(),
    ])


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    from pathlib import Path

    doc = generate_full_document()

    if "--save" in sys.argv:
        out = Path(__file__).parent.parent.parent / "docs" / "workflow_diagram.md"
        out.write_text(doc, encoding="utf-8")
        print(f"Saved to {out}")
    else:
        print(doc)
