"""
Unit tests for the ResearchAgent and the three nodes it feeds.

All tests are fully isolated — no Ollama required.
The LLM is replaced by a mock that returns pre-built ResearchOutput instances.

Test groups
-----------
TestCitationOutput      — Pydantic schema validation for citations
TestResearchOutput      — Pydantic schema validation for the full output
TestBuildContext        — _build_context() prompt-building helper
TestOverrideCitation    — _override_citation_scores() post-processing
TestResearchAgent       — generate(), retry behaviour, error handling
TestResearchNode        — research_node() state contract
TestGeneratorNode       — generator_node() state contract
TestStructuredOutputNode — structured_output_node() state contract
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.agents.research_agent import (
    CitationOutput,
    ResearchAgent,
    ResearchOutput,
    _build_context,
    _override_citation_scores,
)
from app.core.exceptions import LLMError
from app.graph.nodes.generator import generator_node
from app.graph.nodes.research import research_node
from app.graph.nodes.structured_output import structured_output_node
from app.graph.state import AppState, RankedDocument, initial_state


# ── Factories ──────────────────────────────────────────────────────────────────


def _ranked_doc(
    doc_id: str = "d1",
    content: str = "Transformers use self-attention to process sequences in parallel.",
    source: str = "papers/attention.pdf",
    rerank_score: float = 0.92,
) -> RankedDocument:
    return RankedDocument(
        id=doc_id,
        content=content,
        source=source,
        metadata={"page": "1"},
        retrieval_score=0.85,
        rerank_score=rerank_score,
    )


def _citation(
    document_id: str = "d1",
    source: str = "papers/attention.pdf",
    excerpt: str = "Transformers use self-attention",
    rerank_score: float = 0.92,
) -> CitationOutput:
    return CitationOutput(
        document_id=document_id,
        source=source,
        excerpt=excerpt,
        rerank_score=rerank_score,
    )


def _research_output(
    summary: str = "Transformers are neural networks using attention.",
    answer: str = "Transformers rely on self-attention mechanisms [d1] to encode "
                  "contextual relationships without recurrence.",
    citations: list[CitationOutput] | None = None,
) -> ResearchOutput:
    return ResearchOutput(
        summary=summary,
        answer=answer,
        citations=citations if citations is not None else [_citation()],
    )


def _mock_llm(output: ResearchOutput) -> MagicMock:
    """Return a mock LLM whose chain.ainvoke always returns *output*."""
    chain = AsyncMock()
    chain.ainvoke.return_value = output
    llm = MagicMock()
    llm.with_structured_output.return_value = chain
    return llm


def _state(query: str = "How do transformers work?") -> AppState:
    return initial_state(session_id="test-sess", query=query)


def _state_with_docs(docs: list[RankedDocument] | None = None) -> AppState:
    s = _state()
    s["reranked_documents"] = docs if docs is not None else [_ranked_doc()]
    return s


def _state_with_draft(output: ResearchOutput | None = None) -> AppState:
    s = _state()
    s["draft_response"] = (output or _research_output()).model_dump_json()
    return s


# ── CitationOutput ────────────────────────────────────────────────────────────


class TestCitationOutput:
    def test_accepts_valid_citation(self):
        c = _citation()
        assert c.document_id == "d1"
        assert c.rerank_score == pytest.approx(0.92)

    def test_rejects_excerpt_shorter_than_five_chars(self):
        with pytest.raises(ValidationError):
            CitationOutput(document_id="d1", source="s", excerpt="hi", rerank_score=0.5)

    def test_rejects_score_above_one(self):
        with pytest.raises(ValidationError):
            CitationOutput(document_id="d1", source="s", excerpt="long enough excerpt", rerank_score=1.1)

    def test_rejects_score_below_zero(self):
        with pytest.raises(ValidationError):
            CitationOutput(document_id="d1", source="s", excerpt="long enough excerpt", rerank_score=-0.1)

    def test_score_defaults_to_zero_when_not_provided(self):
        c = CitationOutput(document_id="d1", source="s", excerpt="long enough excerpt")
        assert c.rerank_score == pytest.approx(0.0)

    def test_accepts_score_at_boundaries(self):
        c1 = CitationOutput(document_id="d", source="s", excerpt="long enough here", rerank_score=0.0)
        c2 = CitationOutput(document_id="d", source="s", excerpt="long enough here", rerank_score=1.0)
        assert c1.rerank_score == pytest.approx(0.0)
        assert c2.rerank_score == pytest.approx(1.0)


# ── ResearchOutput ────────────────────────────────────────────────────────────


class TestResearchOutput:
    def test_accepts_valid_output(self):
        r = _research_output()
        assert len(r.citations) == 1

    def test_rejects_summary_shorter_than_ten_chars(self):
        with pytest.raises(ValidationError):
            ResearchOutput(summary="Too short", answer="a" * 20, citations=[])

    def test_rejects_answer_shorter_than_twenty_chars(self):
        with pytest.raises(ValidationError):
            ResearchOutput(summary="a" * 10, answer="Too short", citations=[])

    def test_accepts_empty_citations_list(self):
        r = ResearchOutput(summary="a" * 10, answer="a" * 20, citations=[])
        assert r.citations == []

    def test_serialises_and_deserialises_via_json(self):
        original = _research_output()
        restored = ResearchOutput.model_validate_json(original.model_dump_json())
        assert restored.summary == original.summary
        assert restored.answer == original.answer
        assert len(restored.citations) == len(original.citations)

    def test_citation_scores_present_in_json(self):
        r = _research_output()
        data = json.loads(r.model_dump_json())
        assert data["citations"][0]["rerank_score"] == pytest.approx(0.92)


# ── _build_context ────────────────────────────────────────────────────────────


class TestBuildContext:
    def test_includes_document_id(self):
        ctx = _build_context([_ranked_doc(doc_id="abc-123")])
        assert "abc-123" in ctx

    def test_includes_source(self):
        ctx = _build_context([_ranked_doc(source="wiki/ml.txt")])
        assert "wiki/ml.txt" in ctx

    def test_includes_content(self):
        ctx = _build_context([_ranked_doc(content="unique content string xyz")])
        assert "unique content string xyz" in ctx

    def test_includes_rerank_score(self):
        ctx = _build_context([_ranked_doc(rerank_score=0.8765)])
        assert "0.8765" in ctx

    def test_numbered_from_one(self):
        ctx = _build_context([_ranked_doc("d1"), _ranked_doc("d2")])
        assert "Document [1]" in ctx
        assert "Document [2]" in ctx

    def test_handles_empty_documents(self):
        ctx = _build_context([])
        assert "no documents" in ctx.lower()

    def test_multiple_documents_separated(self):
        docs = [_ranked_doc(f"d{i}") for i in range(3)]
        ctx = _build_context(docs)
        # All three IDs must be present
        for i in range(3):
            assert f"d{i}" in ctx

    def test_single_document_context(self):
        ctx = _build_context([_ranked_doc()])
        assert "Document [1]" in ctx
        assert "Document [2]" not in ctx


# ── _override_citation_scores ─────────────────────────────────────────────────


class TestOverrideCitationScores:
    def test_replaces_score_with_actual_document_score(self):
        doc = _ranked_doc(doc_id="d1", rerank_score=0.95)
        output = _research_output(citations=[_citation(document_id="d1", rerank_score=0.0)])
        patched = _override_citation_scores(output, [doc])
        assert patched.citations[0].rerank_score == pytest.approx(0.95)

    def test_keeps_original_score_when_document_not_found(self):
        doc = _ranked_doc(doc_id="d1", rerank_score=0.88)
        output = _research_output(citations=[_citation(document_id="unknown", rerank_score=0.5)])
        patched = _override_citation_scores(output, [doc])
        assert patched.citations[0].rerank_score == pytest.approx(0.5)

    def test_does_not_mutate_original_output(self):
        doc = _ranked_doc(doc_id="d1", rerank_score=0.99)
        output = _research_output(citations=[_citation(document_id="d1", rerank_score=0.1)])
        _override_citation_scores(output, [doc])
        assert output.citations[0].rerank_score == pytest.approx(0.1)

    def test_handles_multiple_citations(self):
        docs = [_ranked_doc(f"d{i}", rerank_score=0.9 - i * 0.1) for i in range(3)]
        citations = [_citation(f"d{i}", rerank_score=0.0) for i in range(3)]
        output = _research_output(citations=citations)
        patched = _override_citation_scores(output, docs)
        for i, c in enumerate(patched.citations):
            assert c.rerank_score == pytest.approx(0.9 - i * 0.1)

    def test_empty_citations_list_unchanged(self):
        output = _research_output(citations=[])
        patched = _override_citation_scores(output, [_ranked_doc()])
        assert patched.citations == []


# ── ResearchAgent ─────────────────────────────────────────────────────────────


class TestResearchAgent:
    @pytest.mark.asyncio
    async def test_returns_research_output_instance(self):
        expected = _research_output()
        agent = ResearchAgent(llm=_mock_llm(expected))
        result = await agent.generate("how do transformers work?", [_ranked_doc()])
        assert isinstance(result, ResearchOutput)

    @pytest.mark.asyncio
    async def test_summary_and_answer_preserved(self):
        expected = _research_output(summary="a" * 10, answer="b" * 25)
        agent = ResearchAgent(llm=_mock_llm(expected))
        result = await agent.generate("query", [_ranked_doc()])
        assert result.summary == expected.summary
        assert result.answer == expected.answer

    @pytest.mark.asyncio
    async def test_citation_scores_overridden_with_document_scores(self):
        doc = _ranked_doc(doc_id="d1", rerank_score=0.97)
        llm_output = _research_output(
            citations=[_citation(document_id="d1", rerank_score=0.0)]
        )
        agent = ResearchAgent(llm=_mock_llm(llm_output))
        result = await agent.generate("query", [doc])
        assert result.citations[0].rerank_score == pytest.approx(0.97)

    @pytest.mark.asyncio
    async def test_handles_empty_documents_list(self):
        expected = ResearchOutput(
            summary="Insufficient information available.",
            answer="The provided documents do not contain information about this query.",
            citations=[],
        )
        agent = ResearchAgent(llm=_mock_llm(expected))
        result = await agent.generate("query", [])
        assert result.citations == []

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_chain_exception(self):
        chain = AsyncMock()
        chain.ainvoke.side_effect = RuntimeError("connection refused")
        llm = MagicMock()
        llm.with_structured_output.return_value = chain
        agent = ResearchAgent(llm=llm)
        with pytest.raises(LLMError, match="Research generation failed"):
            await agent.generate("query", [_ranked_doc()])

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_wrong_return_type(self):
        chain = AsyncMock()
        chain.ainvoke.return_value = {"summary": "...", "answer": "..."}
        llm = MagicMock()
        llm.with_structured_output.return_value = chain
        agent = ResearchAgent(llm=llm)
        with pytest.raises(LLMError, match="Expected ResearchOutput"):
            await agent.generate("query", [_ranked_doc()])

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_none_return(self):
        chain = AsyncMock()
        chain.ainvoke.return_value = None
        llm = MagicMock()
        llm.with_structured_output.return_value = chain
        agent = ResearchAgent(llm=llm)
        with pytest.raises(LLMError):
            await agent.generate("query", [_ranked_doc()])

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self):
        expected = _research_output()
        chain = AsyncMock()
        chain.ainvoke.side_effect = [RuntimeError("timeout"), expected]
        llm = MagicMock()
        llm.with_structured_output.return_value = chain
        agent = ResearchAgent(llm=llm)
        result = await agent.generate("query", [_ranked_doc()])
        assert isinstance(result, ResearchOutput)
        assert chain.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self):
        chain = AsyncMock()
        chain.ainvoke.side_effect = RuntimeError("always fails")
        llm = MagicMock()
        llm.with_structured_output.return_value = chain
        agent = ResearchAgent(llm=llm)
        with pytest.raises(LLMError):
            await agent.generate("query", [_ranked_doc()])
        assert chain.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_context_passed_to_llm_contains_document_id(self):
        doc = _ranked_doc(doc_id="special-id-xyz")
        chain = AsyncMock()
        chain.ainvoke.return_value = _research_output()
        llm = MagicMock()
        llm.with_structured_output.return_value = chain
        agent = ResearchAgent(llm=llm)
        await agent.generate("query", [doc])
        call_messages = chain.ainvoke.call_args[0][0]
        full_text = " ".join(m.content for m in call_messages)
        assert "special-id-xyz" in full_text


# ── research_node ─────────────────────────────────────────────────────────────


class TestResearchNode:
    @pytest.mark.asyncio
    async def test_sets_current_node(self):
        update = await research_node(_state())
        assert update["current_node"] == "research"

    @pytest.mark.asyncio
    async def test_increments_step_count_from_zero(self):
        update = await research_node(_state())
        assert update["step_count"] == 1

    @pytest.mark.asyncio
    async def test_increments_step_count_from_existing(self):
        s = _state()
        s["step_count"] = 3
        update = await research_node(s)
        assert update["step_count"] == 4

    @pytest.mark.asyncio
    async def test_no_errors_key(self):
        update = await research_node(_state())
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_does_not_modify_query(self):
        update = await research_node(_state("my query"))
        assert "query" not in update


# ── generator_node ────────────────────────────────────────────────────────────


class TestGeneratorNode:
    @pytest.mark.asyncio
    async def test_stores_draft_as_json_string(self):
        output = _research_output()
        mock_agent = AsyncMock()
        mock_agent.generate.return_value = output
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_agent):
            update = await generator_node(_state_with_docs())
        assert isinstance(update["draft_response"], str)

    @pytest.mark.asyncio
    async def test_draft_is_valid_research_output_json(self):
        output = _research_output()
        mock_agent = AsyncMock()
        mock_agent.generate.return_value = output
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_agent):
            update = await generator_node(_state_with_docs())
        parsed = ResearchOutput.model_validate_json(update["draft_response"])
        assert parsed.summary == output.summary

    @pytest.mark.asyncio
    async def test_sets_current_node(self):
        mock_agent = AsyncMock()
        mock_agent.generate.return_value = _research_output()
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_agent):
            update = await generator_node(_state_with_docs())
        assert update["current_node"] == "generator"

    @pytest.mark.asyncio
    async def test_increments_step_count(self):
        s = _state_with_docs()
        s["step_count"] = 5
        mock_agent = AsyncMock()
        mock_agent.generate.return_value = _research_output()
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_agent):
            update = await generator_node(s)
        assert update["step_count"] == 6

    @pytest.mark.asyncio
    async def test_passes_query_and_documents_to_agent(self):
        docs = [_ranked_doc("d1"), _ranked_doc("d2")]
        s = _state("what is attention?")
        s["reranked_documents"] = docs
        mock_agent = AsyncMock()
        mock_agent.generate.return_value = _research_output()
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_agent):
            await generator_node(s)
        mock_agent.generate.assert_called_once_with(
            query="what is attention?",
            documents=docs,
        )

    @pytest.mark.asyncio
    async def test_records_error_on_llm_failure(self):
        mock_agent = AsyncMock()
        mock_agent.generate.side_effect = LLMError("ollama crashed")
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_agent):
            update = await generator_node(_state_with_docs())
        assert len(update["errors"]) == 1
        assert update["errors"][0]["node"] == "generator"
        assert "ollama crashed" in update["errors"][0]["message"]
        assert "draft_response" not in update

    @pytest.mark.asyncio
    async def test_records_error_on_unexpected_exception(self):
        mock_agent = AsyncMock()
        mock_agent.generate.side_effect = MemoryError("OOM")
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_agent):
            update = await generator_node(_state_with_docs())
        assert len(update["errors"]) == 1

    @pytest.mark.asyncio
    async def test_no_errors_key_on_success(self):
        mock_agent = AsyncMock()
        mock_agent.generate.return_value = _research_output()
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_agent):
            update = await generator_node(_state_with_docs())
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_empty_reranked_documents_treated_as_empty_list(self):
        s = _state()  # no reranked_documents key
        mock_agent = AsyncMock()
        mock_agent.generate.return_value = _research_output()
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_agent):
            update = await generator_node(s)
        _, kwargs = mock_agent.generate.call_args
        assert kwargs["documents"] == []


# ── structured_output_node ────────────────────────────────────────────────────


class TestStructuredOutputNode:
    @pytest.mark.asyncio
    async def test_populates_structured_output(self):
        update = await structured_output_node(_state_with_draft())
        assert "structured_output" in update
        so = update["structured_output"]
        assert "summary" in so
        assert "answer" in so
        assert "citations" in so

    @pytest.mark.asyncio
    async def test_summary_matches_draft(self):
        output = _research_output(summary="Ten chars ok!")
        update = await structured_output_node(_state_with_draft(output))
        assert update["structured_output"]["summary"] == output.summary

    @pytest.mark.asyncio
    async def test_answer_matches_draft(self):
        output = _research_output(answer="Answer is long enough to pass validation here.")
        update = await structured_output_node(_state_with_draft(output))
        assert update["structured_output"]["answer"] == output.answer

    @pytest.mark.asyncio
    async def test_citations_converted_to_typeddict(self):
        c = _citation(document_id="cite-01", source="src.txt", excerpt="a short quote here", rerank_score=0.88)
        output = _research_output(citations=[c])
        update = await structured_output_node(_state_with_draft(output))
        citations = update["structured_output"]["citations"]
        assert len(citations) == 1
        assert citations[0]["document_id"] == "cite-01"
        assert citations[0]["source"] == "src.txt"
        assert citations[0]["rerank_score"] == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_sets_current_node(self):
        update = await structured_output_node(_state_with_draft())
        assert update["current_node"] == "structured_output"

    @pytest.mark.asyncio
    async def test_increments_step_count(self):
        s = _state_with_draft()
        s["step_count"] = 4
        update = await structured_output_node(s)
        assert update["step_count"] == 5

    @pytest.mark.asyncio
    async def test_records_error_when_draft_is_missing(self):
        s = _state()  # no draft_response
        update = await structured_output_node(s)
        assert len(update["errors"]) == 1
        assert "No draft_response" in update["errors"][0]["message"]
        assert "structured_output" not in update

    @pytest.mark.asyncio
    async def test_records_error_when_draft_is_none(self):
        s = _state()
        s["draft_response"] = None
        update = await structured_output_node(s)
        assert len(update["errors"]) == 1

    @pytest.mark.asyncio
    async def test_records_error_on_invalid_json(self):
        s = _state()
        s["draft_response"] = '{"summary": "ok", "answer": "ok"'  # truncated
        update = await structured_output_node(s)
        assert len(update["errors"]) == 1
        assert "structured_output" not in update

    @pytest.mark.asyncio
    async def test_records_error_on_schema_violation(self):
        s = _state()
        s["draft_response"] = json.dumps({"summary": "short", "answer": "short"})
        update = await structured_output_node(s)
        assert len(update["errors"]) == 1

    @pytest.mark.asyncio
    async def test_no_errors_key_on_success(self):
        update = await structured_output_node(_state_with_draft())
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_empty_citations_preserved(self):
        output = ResearchOutput(
            summary="a" * 10,
            answer="a" * 20,
            citations=[],
        )
        update = await structured_output_node(_state_with_draft(output))
        assert update["structured_output"]["citations"] == []
