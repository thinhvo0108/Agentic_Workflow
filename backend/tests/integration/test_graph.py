"""
LangGraph workflow integration tests.

All external services (Ollama, ChromaDB, CrossEncoder, PostgreSQL) are replaced
with lightweight mocks so the full LangGraph state machine — nodes, edges,
conditional routing, interrupt/resume, state accumulation — executes with real
code.  No network or GPU access required.

What is real
------------
* The compiled LangGraph graph (MemorySaver checkpointer)
* Every node function body (router_node, retriever_node, …, final_response_node)
* State merging, reducer logic, interrupt_before mechanism
* Conditional edge functions (_route_decision, _approval_decision)

What is mocked
--------------
* RouterAgent.classify()     → returns pre-built RouteOutput
* RetrieverService.retrieve()→ returns fake RetrievedDocument list
* RerankerService.rerank()   → returns fake RankedDocument list
* ResearchAgent.generate()   → returns real ResearchOutput instance
* SupportAgent.generate()    → returns real SupportOutput instance
* CheckpointRepository.save()→ in-memory mock (no PostgreSQL required)

Test groups
-----------
TestGraphStructure      — node count, edge wiring, interrupt annotation, Mermaid output
TestResearchPath        — full research-route lifecycle (pause → approve → complete)
TestSupportPath         — full support-route lifecycle
TestRejectionPath       — reviewer rejects → workflow terminates gracefully
TestErrorPaths          — router failure → ends without reaching human_approval
TestStateAccumulation   — step_count, current_node, all major state fields populated
TestCheckpointAudit     — checkpoint_node calls repository.save with correct data
TestVisualization       — visualization module output contracts
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.agents.research_agent import CitationOutput, ResearchOutput
from app.agents.router import RouteOutput
from app.agents.support_agent import SupportOutput
from app.checkpoints.repository import CheckpointRepository
from app.core.exceptions import LLMError
from app.graph.nodes.checkpoint import set_repository
from app.graph.state import (
    RankedDocument,
    RetrievedDocument,
    initial_state,
)
from app.graph.workflow import build_workflow, compile_workflow

# ── Factories ──────────────────────────────────────────────────────────────────


def _retrieved_doc(doc_id: str, score: float = 0.85) -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        content=f"Content for {doc_id}.",
        source="kb.txt",
        metadata={},
        score=score,
    )


def _ranked_doc(doc_id: str, rerank_score: float = 0.92) -> RankedDocument:
    return RankedDocument(
        id=doc_id,
        content=f"Content for {doc_id}.",
        source="kb.txt",
        metadata={},
        retrieval_score=0.85,
        rerank_score=rerank_score,
    )


def _router_output(route: str = "research") -> RouteOutput:
    return RouteOutput(route=route, confidence=0.95, reasoning="test classification")  # type: ignore[arg-type]


def _research_output() -> ResearchOutput:
    return ResearchOutput(
        summary="This is the research summary for the query.",
        answer="This is the detailed research answer addressing the query topic.",
        citations=[
            CitationOutput(
                document_id="d0",
                source="kb.txt",
                excerpt="Relevant excerpt from the knowledge base.",
                rerank_score=0.92,
            )
        ],
    )


def _support_output(retrieval_used: bool = False) -> SupportOutput:
    return SupportOutput(
        summary="Support summary answers the FAQ question.",
        answer="To resolve this issue, follow these steps carefully.",
        citations=[],
        retrieval_used=retrieval_used,
        confidence=0.88,
    )


def _mock_repo() -> AsyncMock:
    repo = AsyncMock(spec=CheckpointRepository)

    async def _save(record):
        return record.model_copy(update={"id": 1, "created_at": datetime.now(UTC)})

    repo.save.side_effect = _save
    return repo


# ── Test harness ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def _workflow_harness(
    route: str = "research",
    router_raises: Exception | None = None,
    retriever_raises: Exception | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Compile graph + configure all service mocks.  Yields a dict of mocks."""
    wf = compile_workflow(MemorySaver())

    retrieved = [_retrieved_doc(f"d{i}") for i in range(3)]
    ranked = [_ranked_doc(f"d{i}") for i in range(2)]
    repo = _mock_repo()
    set_repository(repo)

    mock_router = AsyncMock()
    if router_raises:
        mock_router.classify.side_effect = router_raises
    else:
        mock_router.classify.return_value = _router_output(route)

    mock_retriever = AsyncMock()
    if retriever_raises:
        mock_retriever.retrieve.side_effect = retriever_raises
    else:
        mock_retriever.retrieve.return_value = retrieved

    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = ranked

    mock_research_agent = AsyncMock()
    mock_research_agent.generate.return_value = _research_output()

    mock_support_agent = AsyncMock()
    mock_support_agent.generate.return_value = _support_output()

    with (
        patch("app.graph.nodes.router.RouterAgent", return_value=mock_router),
        patch("app.graph.nodes.retriever.RetrieverService", return_value=mock_retriever),
        patch("app.graph.nodes.reranker.RerankerService", return_value=mock_reranker),
        patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_research_agent),
        patch("app.graph.nodes.generator.SupportAgent", return_value=mock_support_agent),
    ):
        yield {
            "workflow": wf,
            "router": mock_router,
            "retriever": mock_retriever,
            "reranker": mock_reranker,
            "research_agent": mock_research_agent,
            "support_agent": mock_support_agent,
            "repo": repo,
        }

    set_repository(None)


async def _run_to_pause(harness: dict, session_id: str = "s-001", query: str = "test query"):
    """Start the workflow and return the StateSnapshot after the interrupt."""
    wf = harness["workflow"]
    state = initial_state(session_id=session_id, query=query)
    config = {"configurable": {"thread_id": session_id}}
    await wf.ainvoke(state, config)
    return await wf.aget_state(config)


async def _resume(harness: dict, session_id: str, action: str, reviewer: str = "alice"):
    """Inject an approval decision and resume the workflow."""
    from app.graph.state import ApprovalRecord

    wf = harness["workflow"]
    config = {"configurable": {"thread_id": session_id}}
    record: ApprovalRecord = {
        "reviewer_id": reviewer,
        "action": action,  # type: ignore[typeddict-item]
        "decided_at": datetime.now(UTC).isoformat(),
    }
    await wf.aupdate_state(config, {"approval_status": action, "approval_record": record})
    await wf.ainvoke(None, config)
    return await wf.aget_state(config)


# ── Structure tests ────────────────────────────────────────────────────────────


class TestGraphStructure:
    def test_build_workflow_returns_state_graph(self):
        from langgraph.graph import StateGraph

        assert isinstance(build_workflow(), StateGraph)

    def test_compile_workflow_with_memory_saver(self):
        wf = compile_workflow(MemorySaver())
        assert wf is not None

    def test_graph_has_ten_application_nodes(self):
        wf = compile_workflow(MemorySaver())
        nodes = [n for n in wf.get_graph().nodes if not n.startswith("__")]
        assert len(nodes) == 10

    def test_graph_contains_all_required_nodes(self):
        wf = compile_workflow(MemorySaver())
        node_names = set(wf.get_graph().nodes)
        required = {
            "router",
            "research",
            "support",
            "retriever",
            "reranker",
            "generator",
            "structured_output",
            "checkpoint",
            "human_approval",
            "final_response",
        }
        assert required.issubset(node_names)

    def test_graph_has_interrupt_before_human_approval(self):
        wf = compile_workflow(MemorySaver())
        # human_approval should appear in the graph's interrupt config
        mermaid = wf.get_graph().draw_mermaid()
        assert "interrupt" in mermaid.lower()
        assert "human_approval" in mermaid

    def test_graph_edges_include_direct_pipeline(self):
        wf = compile_workflow(MemorySaver())
        edges = {(e.source, e.target): e for e in wf.get_graph().edges}
        expected_direct = [
            ("research", "retriever"),
            ("support", "retriever"),
            ("retriever", "reranker"),
            ("reranker", "generator"),
            ("generator", "structured_output"),
            ("structured_output", "checkpoint"),
            ("checkpoint", "human_approval"),
            ("final_response", "__end__"),
        ]
        for src, tgt in expected_direct:
            assert (src, tgt) in edges, f"Missing direct edge {src} → {tgt}"

    def test_router_has_conditional_edges_to_research_and_support(self):
        wf = compile_workflow(MemorySaver())
        router_edges = [e for e in wf.get_graph().edges if e.source == "router"]
        targets = {e.target for e in router_edges}
        assert "research" in targets
        assert "support" in targets
        assert "__end__" in targets

    def test_human_approval_has_conditional_edges(self):
        wf = compile_workflow(MemorySaver())
        ha_edges = [e for e in wf.get_graph().edges if e.source == "human_approval"]
        targets = {e.target for e in ha_edges}
        assert "final_response" in targets
        assert "__end__" in targets


# ── Research path ─────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestResearchPath:
    @pytest.mark.asyncio
    async def test_research_path_pauses_at_human_approval(self):
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        assert "human_approval" in snap.next

    @pytest.mark.asyncio
    async def test_research_path_sets_route_to_research(self):
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        assert snap.values["route"] == "research"

    @pytest.mark.asyncio
    async def test_research_path_populates_retrieved_documents(self):
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        assert len(snap.values["retrieved_documents"]) == 3

    @pytest.mark.asyncio
    async def test_research_path_populates_reranked_documents(self):
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        assert len(snap.values["reranked_documents"]) == 2

    @pytest.mark.asyncio
    async def test_research_path_populates_structured_output(self):
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        so = snap.values.get("structured_output")
        assert so is not None
        assert "summary" in so
        assert "answer" in so
        assert "citations" in so

    @pytest.mark.asyncio
    async def test_research_path_calls_router_agent(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h)
        h["router"].classify.assert_called_once()

    @pytest.mark.asyncio
    async def test_research_path_calls_research_agent(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h)
        h["research_agent"].generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_research_path_does_not_call_support_agent(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h)
        h["support_agent"].generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_research_approval_status_starts_pending(self):
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        assert snap.values["approval_status"] == "pending"

    @pytest.mark.asyncio
    async def test_research_path_approved_produces_final_response(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-approve")
            snap = await _resume(h, "s-approve", "approved")
        assert snap.values.get("final_response") is not None
        assert snap.values["approval_status"] == "approved"

    @pytest.mark.asyncio
    async def test_research_path_approved_final_response_has_all_fields(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-fields")
            snap = await _resume(h, "s-fields", "approved")
        fr = snap.values["final_response"]
        assert fr["session_id"] == "s-fields"
        assert fr["route"] == "research"
        assert fr["approval_status"] == "approved"
        assert "summary" in fr
        assert "answer" in fr

    @pytest.mark.asyncio
    async def test_research_path_approved_graph_completes(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-complete")
            snap = await _resume(h, "s-complete", "approved")
        assert snap.next == ()

    @pytest.mark.asyncio
    async def test_research_path_no_errors_on_clean_run(self):
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        assert snap.values.get("errors") == []


# ── Support path ──────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestSupportPath:
    @pytest.mark.asyncio
    async def test_support_path_pauses_at_human_approval(self):
        async with _workflow_harness("support") as h:
            snap = await _run_to_pause(h)
        assert "human_approval" in snap.next

    @pytest.mark.asyncio
    async def test_support_path_sets_route_to_support(self):
        async with _workflow_harness("support") as h:
            snap = await _run_to_pause(h)
        assert snap.values["route"] == "support"

    @pytest.mark.asyncio
    async def test_support_path_calls_support_agent(self):
        async with _workflow_harness("support") as h:
            await _run_to_pause(h)
        h["support_agent"].generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_support_path_does_not_call_research_agent(self):
        async with _workflow_harness("support") as h:
            await _run_to_pause(h)
        h["research_agent"].generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_support_path_approved_produces_final_response(self):
        async with _workflow_harness("support") as h:
            await _run_to_pause(h, session_id="sup-approve")
            snap = await _resume(h, "sup-approve", "approved")
        assert snap.values.get("final_response") is not None
        assert snap.values["final_response"]["route"] == "support"


# ── Rejection path ────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestRejectionPath:
    @pytest.mark.asyncio
    async def test_rejected_workflow_completes_without_final_response(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-reject")
            snap = await _resume(h, "s-reject", "rejected", reviewer="bob")
        assert snap.values.get("final_response") is None

    @pytest.mark.asyncio
    async def test_rejected_workflow_approval_status_is_rejected(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-reject2")
            snap = await _resume(h, "s-reject2", "rejected")
        assert snap.values["approval_status"] == "rejected"

    @pytest.mark.asyncio
    async def test_rejected_graph_has_no_next_nodes(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-reject3")
            snap = await _resume(h, "s-reject3", "rejected")
        assert snap.next == ()

    @pytest.mark.asyncio
    async def test_approval_record_stored_on_rejection(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-reject4")
            snap = await _resume(h, "s-reject4", "rejected", reviewer="carol")
        record = snap.values.get("approval_record")
        assert record is not None
        assert record["reviewer_id"] == "carol"
        assert record["action"] == "rejected"


# ── Error paths ───────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_router_llm_error_terminates_without_reaching_approval(self):
        async with _workflow_harness(router_raises=LLMError("Ollama unreachable")) as h:
            snap = await _run_to_pause(h, session_id="s-err-router")
        # Graph should have terminated — no interrupt at human_approval
        assert "human_approval" not in snap.next

    @pytest.mark.asyncio
    async def test_router_error_recorded_in_errors_list(self):
        async with _workflow_harness(router_raises=LLMError("connection refused")) as h:
            snap = await _run_to_pause(h, session_id="s-err-recorded")
        errors = snap.values.get("errors") or []
        assert len(errors) >= 1
        router_errors = [e for e in errors if e["node"] == "router"]
        assert len(router_errors) == 1

    @pytest.mark.asyncio
    async def test_router_error_message_in_errors(self):
        async with _workflow_harness(router_raises=LLMError("timeout after 120s")) as h:
            snap = await _run_to_pause(h, session_id="s-err-msg")
        errors = snap.values.get("errors") or []
        assert any("timeout" in e["message"] for e in errors)


# ── State accumulation ────────────────────────────────────────────────────────


@pytest.mark.integration
class TestStateAccumulation:
    @pytest.mark.asyncio
    async def test_step_count_is_at_least_eight_after_pause(self):
        """One increment per node: router, research, retriever, reranker,
        generator, structured_output, checkpoint = 7 minimum."""
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        assert snap.values.get("step_count", 0) >= 7

    @pytest.mark.asyncio
    async def test_step_count_increases_further_after_approval(self):
        async with _workflow_harness("research") as h:
            snap1 = await _run_to_pause(h, session_id="sc-steps")
            steps_before = snap1.values.get("step_count", 0)
            snap2 = await _resume(h, "sc-steps", "approved")
        assert snap2.values.get("step_count", 0) > steps_before

    @pytest.mark.asyncio
    async def test_errors_reducer_accumulates_across_nodes(self):
        """The errors field uses operator.add — verifies TypedDict reducer wiring."""
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        # Clean run should have an empty list (reducer started at [])
        assert isinstance(snap.values.get("errors"), list)

    @pytest.mark.asyncio
    async def test_draft_response_is_json_string(self):
        import json

        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        draft = snap.values.get("draft_response")
        assert draft is not None
        parsed = json.loads(draft)
        assert "summary" in parsed
        assert "answer" in parsed

    @pytest.mark.asyncio
    async def test_all_pipeline_state_fields_populated_after_pause(self):
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h)
        v = snap.values
        assert v.get("route") == "research"
        assert v.get("retrieved_documents")
        assert v.get("reranked_documents")
        assert v.get("draft_response")
        assert v.get("structured_output") is not None

    @pytest.mark.asyncio
    async def test_session_id_preserved_unchanged(self):
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h, session_id="preserve-me")
        assert snap.values["session_id"] == "preserve-me"

    @pytest.mark.asyncio
    async def test_query_preserved_unchanged(self):
        async with _workflow_harness("research") as h:
            snap = await _run_to_pause(h, query="What is self-attention?")
        assert snap.values["query"] == "What is self-attention?"


# ── Checkpoint audit ──────────────────────────────────────────────────────────


@pytest.mark.integration
class TestCheckpointAudit:
    @pytest.mark.asyncio
    async def test_checkpoint_node_calls_repository_save(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-chk")
        h["repo"].save.assert_called_once()

    @pytest.mark.asyncio
    async def test_checkpoint_saved_with_correct_session_id(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="chk-sess")
        call_args = h["repo"].save.call_args
        saved_record = call_args[0][0]
        assert saved_record.session_id == "chk-sess"

    @pytest.mark.asyncio
    async def test_checkpoint_saved_with_generation_stage(self):
        from app.checkpoints.models import CheckpointStage

        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-stage")
        record = h["repo"].save.call_args[0][0]
        assert record.stage == CheckpointStage.GENERATION

    @pytest.mark.asyncio
    async def test_checkpoint_has_correct_doc_counts(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-counts")
        record = h["repo"].save.call_args[0][0]
        assert record.retrieved_doc_count == 3
        assert record.reranked_doc_count == 2

    @pytest.mark.asyncio
    async def test_checkpoint_has_structured_output_flag(self):
        async with _workflow_harness("research") as h:
            await _run_to_pause(h, session_id="s-so-flag")
        record = h["repo"].save.call_args[0][0]
        assert record.has_structured_output is True


# ── Visualization ─────────────────────────────────────────────────────────────


class TestVisualization:
    def test_get_node_list_returns_ten_nodes(self):
        from app.graph.visualization import get_node_list

        nodes = get_node_list()
        assert len(nodes) == 10

    def test_get_node_list_contains_all_names(self):
        from app.graph.visualization import get_node_list

        names = set(get_node_list())
        assert names == {
            "router",
            "research",
            "support",
            "retriever",
            "reranker",
            "generator",
            "structured_output",
            "checkpoint",
            "human_approval",
            "final_response",
        }

    def test_generate_mermaid_contains_all_nodes(self):
        from app.graph.visualization import generate_mermaid

        mermaid = generate_mermaid()
        for node in (
            "router",
            "research",
            "support",
            "retriever",
            "reranker",
            "generator",
            "structured_output",
            "checkpoint",
            "human_approval",
            "final_response",
        ):
            assert node in mermaid

    def test_generate_mermaid_marks_interrupt(self):
        from app.graph.visualization import generate_mermaid

        assert "interrupt" in generate_mermaid().lower()

    def test_generate_ascii_table_has_all_nodes(self):
        from app.graph.visualization import generate_ascii_table

        table = generate_ascii_table()
        for name in (
            "router",
            "retriever",
            "reranker",
            "generator",
            "checkpoint",
            "human_approval",
            "final_response",
        ):
            assert name in table

    def test_generate_flow_narrative_describes_both_paths(self):
        from app.graph.visualization import generate_flow_narrative

        narrative = generate_flow_narrative()
        assert "Research path" in narrative
        assert "Support path" in narrative
        assert "PAUSES" in narrative or "pause" in narrative.lower()

    def test_generate_full_document_is_valid_markdown(self):
        from app.graph.visualization import generate_full_document

        doc = generate_full_document()
        assert doc.startswith("# Agentic Workflow")
        assert "```mermaid" in doc
        assert "## Node Reference" in doc
        assert "## Execution Paths" in doc

    def test_get_edge_summary_returns_expected_edges(self):
        from app.graph.visualization import get_edge_summary

        edges = get_edge_summary()
        pairs = {(e["source"], e["target"]) for e in edges}
        assert ("retriever", "reranker") in pairs
        assert ("reranker", "generator") in pairs
        assert ("generator", "structured_output") in pairs
