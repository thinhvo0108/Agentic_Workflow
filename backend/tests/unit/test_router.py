"""
Unit tests for RouterAgent and router_node.

All tests are fully isolated — no Ollama process is required.
The LLM is replaced by a mock that returns pre-built RouteOutput instances,
so tests run quickly and deterministically.

Test groups
-----------
TestRouteOutput       — Pydantic schema validation (no I/O at all)
TestRouterAgentClassify — RouterAgent.classify() with a mock chain
TestRouterAgentRetry    — retry behaviour on transient LLM failures
TestRouterNode          — router_node() integration with a patched RouterAgent
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.agents.router import RouteOutput, RouterAgent
from app.core.exceptions import LLMError
from app.graph.nodes.router import router_node
from app.graph.state import initial_state


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_llm(route: str, confidence: float = 0.92) -> MagicMock:
    """Return a mock LLM whose chain.ainvoke always returns a RouteOutput."""
    output = RouteOutput(route=route, confidence=confidence, reasoning="mock reasoning")
    chain = AsyncMock()
    chain.ainvoke.return_value = output
    llm = MagicMock()
    llm.with_structured_output.return_value = chain
    return llm


def _state(query: str = "test query") -> dict:
    return initial_state(session_id="sess-001", query=query)


# ── RouteOutput schema ─────────────────────────────────────────────────────────


class TestRouteOutput:
    def test_accepts_research(self):
        r = RouteOutput(route="research", confidence=0.9, reasoning="technical")
        assert r.route == "research"

    def test_accepts_support(self):
        r = RouteOutput(route="support", confidence=0.75, reasoning="operational")
        assert r.route == "support"

    def test_rejects_unknown_route(self):
        with pytest.raises(ValidationError):
            RouteOutput(route="unknown", confidence=0.5, reasoning="x")  # type: ignore[arg-type]

    def test_rejects_confidence_above_one(self):
        with pytest.raises(ValidationError):
            RouteOutput(route="research", confidence=1.1, reasoning="x")

    def test_rejects_confidence_below_zero(self):
        with pytest.raises(ValidationError):
            RouteOutput(route="support", confidence=-0.1, reasoning="x")

    def test_rejects_empty_reasoning(self):
        with pytest.raises(ValidationError):
            RouteOutput(route="research", confidence=0.8, reasoning="")


# ── RouterAgent.classify ───────────────────────────────────────────────────────


class TestRouterAgentClassify:
    @pytest.mark.asyncio
    async def test_returns_research_for_technical_query(self):
        agent = RouterAgent(llm=_make_llm("research"))
        result = await agent.classify("How does RAFT consensus achieve fault tolerance?")
        assert result.route == "research"

    @pytest.mark.asyncio
    async def test_returns_support_for_operational_query(self):
        agent = RouterAgent(llm=_make_llm("support"))
        result = await agent.classify("I can't log in to my account.")
        assert result.route == "support"

    @pytest.mark.asyncio
    async def test_result_has_all_fields(self):
        agent = RouterAgent(llm=_make_llm("research", confidence=0.87))
        result = await agent.classify("explain microservices")
        assert result.route == "research"
        assert result.confidence == pytest.approx(0.87)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    @pytest.mark.asyncio
    async def test_raises_llm_error_when_chain_raises(self):
        chain = AsyncMock()
        chain.ainvoke.side_effect = RuntimeError("connection refused")
        llm = MagicMock()
        llm.with_structured_output.return_value = chain

        agent = RouterAgent(llm=llm)
        with pytest.raises(LLMError, match="Router classification failed"):
            await agent.classify("any query")

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_wrong_return_type(self):
        chain = AsyncMock()
        chain.ainvoke.return_value = {"route": "research"}  # dict, not RouteOutput
        llm = MagicMock()
        llm.with_structured_output.return_value = chain

        agent = RouterAgent(llm=llm)
        with pytest.raises(LLMError, match="Expected RouteOutput"):
            await agent.classify("query")

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_none_return(self):
        chain = AsyncMock()
        chain.ainvoke.return_value = None
        llm = MagicMock()
        llm.with_structured_output.return_value = chain

        agent = RouterAgent(llm=llm)
        with pytest.raises(LLMError):
            await agent.classify("query")


# ── RouterAgent retry behaviour ────────────────────────────────────────────────


class TestRouterAgentRetry:
    @pytest.mark.asyncio
    async def test_succeeds_after_one_transient_failure(self):
        """First call fails, second succeeds — result is correct."""
        good = RouteOutput(route="support", confidence=0.8, reasoning="retry ok")
        chain = AsyncMock()
        chain.ainvoke.side_effect = [RuntimeError("timeout"), good]
        llm = MagicMock()
        llm.with_structured_output.return_value = chain

        agent = RouterAgent(llm=llm)
        result = await agent.classify("help me")

        assert result.route == "support"
        assert chain.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self):
        """All three attempts fail — LLMError is eventually raised."""
        chain = AsyncMock()
        chain.ainvoke.side_effect = RuntimeError("always fails")
        llm = MagicMock()
        llm.with_structured_output.return_value = chain

        agent = RouterAgent(llm=llm)
        with pytest.raises(LLMError):
            await agent.classify("query")

        assert chain.ainvoke.call_count == 3  # 3 attempts total


# ── router_node ────────────────────────────────────────────────────────────────


class TestRouterNode:
    @pytest.mark.asyncio
    async def test_sets_route_on_success(self):
        good = RouteOutput(route="research", confidence=0.95, reasoning="technical")
        mock_agent = AsyncMock()
        mock_agent.classify.return_value = good

        with patch("app.graph.nodes.router.RouterAgent", return_value=mock_agent):
            update = await router_node(_state("how does BERT work?"))

        assert update["route"] == "research"

    @pytest.mark.asyncio
    async def test_sets_current_node(self):
        good = RouteOutput(route="support", confidence=0.8, reasoning="faq")
        mock_agent = AsyncMock()
        mock_agent.classify.return_value = good

        with patch("app.graph.nodes.router.RouterAgent", return_value=mock_agent):
            update = await router_node(_state("reset password"))

        assert update["current_node"] == "router"

    @pytest.mark.asyncio
    async def test_increments_step_count_from_zero(self):
        good = RouteOutput(route="support", confidence=0.9, reasoning="ok")
        mock_agent = AsyncMock()
        mock_agent.classify.return_value = good

        with patch("app.graph.nodes.router.RouterAgent", return_value=mock_agent):
            update = await router_node(_state())

        assert update["step_count"] == 1

    @pytest.mark.asyncio
    async def test_increments_step_count_from_existing(self):
        good = RouteOutput(route="research", confidence=0.9, reasoning="ok")
        mock_agent = AsyncMock()
        mock_agent.classify.return_value = good

        state = _state()
        state["step_count"] = 4

        with patch("app.graph.nodes.router.RouterAgent", return_value=mock_agent):
            update = await router_node(state)

        assert update["step_count"] == 5

    @pytest.mark.asyncio
    async def test_records_error_and_omits_route_on_llm_failure(self):
        mock_agent = AsyncMock()
        mock_agent.classify.side_effect = LLMError("ollama unreachable")

        with patch("app.graph.nodes.router.RouterAgent", return_value=mock_agent):
            update = await router_node(_state())

        assert "route" not in update
        assert len(update["errors"]) == 1
        assert update["errors"][0]["node"] == "router"
        assert "ollama unreachable" in update["errors"][0]["message"]

    @pytest.mark.asyncio
    async def test_records_error_on_unexpected_exception(self):
        mock_agent = AsyncMock()
        mock_agent.classify.side_effect = RuntimeError("something exploded")

        with patch("app.graph.nodes.router.RouterAgent", return_value=mock_agent):
            update = await router_node(_state())

        assert len(update["errors"]) == 1
        assert update["errors"][0]["node"] == "router"

    @pytest.mark.asyncio
    async def test_error_contains_iso_timestamp(self):
        mock_agent = AsyncMock()
        mock_agent.classify.side_effect = LLMError("fail")

        with patch("app.graph.nodes.router.RouterAgent", return_value=mock_agent):
            update = await router_node(_state())

        ts = update["errors"][0]["timestamp"]
        assert "T" in ts  # ISO-8601 datetime separator
        assert "+00:00" in ts or ts.endswith("Z")  # UTC marker

    @pytest.mark.asyncio
    async def test_no_errors_list_on_success(self):
        good = RouteOutput(route="research", confidence=0.9, reasoning="ok")
        mock_agent = AsyncMock()
        mock_agent.classify.return_value = good

        with patch("app.graph.nodes.router.RouterAgent", return_value=mock_agent):
            update = await router_node(_state())

        assert "errors" not in update
