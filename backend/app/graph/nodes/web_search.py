"""
Web-search node.

Runs on the manual-approval path (auto_approved=False) between the
auto_approval_gate and human_approval nodes.  Fetches top web results for
the current query via DuckDuckGo (no API key required) and stores them in
state so the /draft endpoint can surface them to the reviewer.

Errors are caught and logged; the node always returns so the workflow
reaches human_approval even if the search call fails.
"""

import asyncio

from app.core.logging import get_logger
from app.graph.state import AppState, WebSearchResult, make_error

_logger = get_logger(__name__)
_NODE = "web_search"
_MAX_RESULTS = 5


def _ddg_search(query: str) -> list[dict]:
    """Synchronous DuckDuckGo text search (run in a thread pool)."""
    from ddgs import DDGS
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=_MAX_RESULTS))


async def web_search_node(state: AppState) -> dict:
    """Fetch top web results for the query and store them in state."""
    step = state.get("step_count", 0) + 1
    query = (state.get("query") or "").strip()

    if not query:
        _logger.warning("web_search_skip_empty_query", session_id=state["session_id"])
        return {"web_search_results": [], "current_node": _NODE, "step_count": step}

    session_id = state["session_id"]

    try:
        raw = await asyncio.to_thread(_ddg_search, query)
        results: list[WebSearchResult] = [
            WebSearchResult(
                title=r.get("title") or "",
                link=r.get("href") or "",
                snippet=r.get("body") or "",
            )
            for r in raw
        ]
        _logger.info(
            "web_search_done",
            session_id=session_id,
            result_count=len(results),
        )
        return {"web_search_results": results, "current_node": _NODE, "step_count": step}

    except Exception as exc:
        _logger.error(
            "web_search_failed",
            session_id=session_id,
            error=str(exc),
        )
        return {
            "web_search_results": [],
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, str(exc))],
        }
