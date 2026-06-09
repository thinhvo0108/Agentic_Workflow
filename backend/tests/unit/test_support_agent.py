"""
Unit tests for the SupportAgent and its supporting nodes.

No Ollama required — every LLM call is replaced by a mock.

Test groups
-----------
TestConfidenceAssessment   — Pydantic schema validation
TestSupportOutput          — Subclass contract + structural compatibility
TestNeedsRetrieval         — _needs_retrieval() pure function
TestSupportAgentGenerate   — Full generate() dispatch logic
TestSupportAgentTriage     — _assess_confidence() pass
TestSupportAgentDirect     — _generate_direct() pass
TestSupportAgentContext    — _generate_with_context() pass
TestSupportAgentRetry      — Retry behaviour across both chains
TestSupportNode            — support_node() state contract
TestGeneratorNodeRouting   — generator_node() route dispatch (research vs support)
"""

import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pydantic import ValidationError

from app.agents.research_agent import CitationOutput, ResearchOutput
from app.agents.support_agent import (
    CONFIDENCE_THRESHOLD,
    ConfidenceAssessment,
    SupportAgent,
    SupportOutput,
    _needs_retrieval,
)
from app.core.exceptions import LLMError
from app.graph.nodes.generator import generator_node
from app.graph.nodes.support import support_node
from app.graph.state import AppState, RankedDocument, initial_state


# ── Factories ──────────────────────────────────────────────────────────────────


def _assessment(
    can_answer: bool = True,
    confidence: float = 0.85,
    answer_type: str = "faq",
    reasoning: str = "Standard FAQ question.",
) -> ConfidenceAssessment:
    return ConfidenceAssessment(
        can_answer_directly=can_answer,
        confidence=confidence,
        answer_type=answer_type,
        reasoning=reasoning,
    )


def _support_output(
    summary: str = "Password reset is straightforward.",
    answer: str = "To reset your password, navigate to Settings and click Reset.",
    citations: list | None = None,
    retrieval_used: bool = False,
    confidence: float = 0.85,
) -> SupportOutput:
    return SupportOutput(
        summary=summary,
        answer=answer,
        citations=citations or [],
        retrieval_used=retrieval_used,
        confidence=confidence,
    )


def _ranked_doc(
    doc_id: str = "kb-001",
    content: str = "Reset password via Settings > Security > Reset.",
    rerank_score: float = 0.91,
) -> RankedDocument:
    return RankedDocument(
        id=doc_id, content=content, source="kb.txt",
        metadata={}, retrieval_score=0.8, rerank_score=rerank_score,
    )


def _mock_llm(
    assessment: ConfidenceAssessment,
    support_out: SupportOutput,
) -> MagicMock:
    """Return a mock LLM whose two chains return *assessment* then *support_out*."""
    confidence_chain = AsyncMock()
    confidence_chain.ainvoke.return_value = assessment

    generate_chain = AsyncMock()
    generate_chain.ainvoke.return_value = support_out

    llm = MagicMock()
    # First with_structured_output call → confidence_chain, second → generate_chain
    llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
    return llm


def _state(query: str = "How do I reset my password?") -> AppState:
    return initial_state(session_id="s-support", query=query)


def _state_with_docs(docs: list | None = None) -> AppState:
    s = _state()
    s["reranked_documents"] = docs if docs is not None else [_ranked_doc()]
    return s


def _state_with_route(route: str, docs: list | None = None) -> AppState:
    s = _state_with_docs(docs)
    s["route"] = route
    return s


# ── ConfidenceAssessment ───────────────────────────────────────────────────────


class TestConfidenceAssessment:
    def test_accepts_faq_type(self):
        a = _assessment(answer_type="faq")
        assert a.answer_type == "faq"

    def test_accepts_troubleshooting_type(self):
        a = _assessment(answer_type="troubleshooting")
        assert a.answer_type == "troubleshooting"

    def test_accepts_general_type(self):
        a = _assessment(answer_type="general")
        assert a.answer_type == "general"

    def test_accepts_requires_context_type(self):
        a = _assessment(answer_type="requires_context")
        assert a.answer_type == "requires_context"

    def test_rejects_invalid_answer_type(self):
        with pytest.raises(ValidationError):
            ConfidenceAssessment(
                can_answer_directly=True,
                confidence=0.8,
                answer_type="unknown",  # type: ignore[arg-type]
                reasoning="test",
            )

    def test_rejects_confidence_above_one(self):
        with pytest.raises(ValidationError):
            ConfidenceAssessment(
                can_answer_directly=True, confidence=1.1,
                answer_type="faq", reasoning="test",
            )

    def test_rejects_confidence_below_zero(self):
        with pytest.raises(ValidationError):
            ConfidenceAssessment(
                can_answer_directly=True, confidence=-0.1,
                answer_type="faq", reasoning="test",
            )

    def test_rejects_short_reasoning(self):
        with pytest.raises(ValidationError):
            ConfidenceAssessment(
                can_answer_directly=True, confidence=0.8,
                answer_type="faq", reasoning="x",
            )

    def test_accepts_boundary_confidence_values(self):
        a0 = _assessment(confidence=0.0)
        a1 = _assessment(confidence=1.0)
        assert a0.confidence == pytest.approx(0.0)
        assert a1.confidence == pytest.approx(1.0)


# ── SupportOutput ─────────────────────────────────────────────────────────────


class TestSupportOutput:
    def test_is_subclass_of_research_output(self):
        assert issubclass(SupportOutput, ResearchOutput)

    def test_isinstance_of_research_output(self):
        so = _support_output()
        assert isinstance(so, ResearchOutput)

    def test_has_retrieval_used_field(self):
        so = _support_output(retrieval_used=True)
        assert so.retrieval_used is True

    def test_has_confidence_field(self):
        so = _support_output(confidence=0.77)
        assert so.confidence == pytest.approx(0.77)

    def test_defaults_retrieval_used_to_false(self):
        so = SupportOutput(
            summary="a" * 10, answer="b" * 20, citations=[]
        )
        assert so.retrieval_used is False

    def test_defaults_confidence_to_zero(self):
        so = SupportOutput(
            summary="a" * 10, answer="b" * 20, citations=[]
        )
        assert so.confidence == pytest.approx(0.0)

    def test_validates_as_research_output_from_json(self):
        """Extra fields must be silently dropped — structured_output_node compatibility."""
        so = _support_output(retrieval_used=True, confidence=0.9)
        parsed = ResearchOutput.model_validate_json(so.model_dump_json())
        assert parsed.summary == so.summary
        assert not hasattr(parsed, "retrieval_used")

    def test_inherits_research_output_validation(self):
        with pytest.raises(ValidationError):
            SupportOutput(summary="short", answer="b" * 20, citations=[])


# ── _needs_retrieval ──────────────────────────────────────────────────────────


class TestNeedsRetrieval:
    def test_high_confidence_faq_does_not_need_retrieval(self):
        a = _assessment(can_answer=True, confidence=0.9, answer_type="faq")
        assert _needs_retrieval(a) is False

    def test_high_confidence_troubleshooting_does_not_need_retrieval(self):
        a = _assessment(can_answer=True, confidence=0.8, answer_type="troubleshooting")
        assert _needs_retrieval(a) is False

    def test_low_confidence_needs_retrieval(self):
        a = _assessment(can_answer=True, confidence=0.5, answer_type="faq")
        assert _needs_retrieval(a) is True

    def test_cannot_answer_directly_needs_retrieval(self):
        a = _assessment(can_answer=False, confidence=0.9, answer_type="general")
        assert _needs_retrieval(a) is True

    def test_requires_context_type_needs_retrieval_regardless_of_confidence(self):
        a = _assessment(can_answer=True, confidence=0.95, answer_type="requires_context")
        assert _needs_retrieval(a) is True

    def test_confidence_exactly_at_threshold_does_not_need_retrieval(self):
        a = _assessment(can_answer=True, confidence=CONFIDENCE_THRESHOLD, answer_type="faq")
        # confidence >= threshold, so no retrieval needed
        assert _needs_retrieval(a) is False

    def test_confidence_just_below_threshold_needs_retrieval(self):
        a = _assessment(can_answer=True, confidence=CONFIDENCE_THRESHOLD - 0.01, answer_type="faq")
        assert _needs_retrieval(a) is True


# ── SupportAgent.generate — dispatch ──────────────────────────────────────────


class TestSupportAgentGenerate:
    @pytest.mark.asyncio
    async def test_direct_path_when_high_confidence(self):
        assessment = _assessment(can_answer=True, confidence=0.9, answer_type="faq")
        output = _support_output()
        agent = SupportAgent(llm=_mock_llm(assessment, output))
        result = await agent.generate("reset password", [_ranked_doc()])
        assert isinstance(result, SupportOutput)

    @pytest.mark.asyncio
    async def test_direct_path_sets_retrieval_used_false(self):
        assessment = _assessment(can_answer=True, confidence=0.9, answer_type="faq")
        agent = SupportAgent(llm=_mock_llm(assessment, _support_output()))
        result = await agent.generate("reset password", [_ranked_doc()])
        assert result.retrieval_used is False

    @pytest.mark.asyncio
    async def test_rag_path_when_low_confidence(self):
        assessment = _assessment(can_answer=True, confidence=0.4, answer_type="general")
        output = _support_output()
        agent = SupportAgent(llm=_mock_llm(assessment, output))
        result = await agent.generate("complex config issue", [_ranked_doc()])
        assert result.retrieval_used is True

    @pytest.mark.asyncio
    async def test_rag_path_when_requires_context(self):
        assessment = _assessment(can_answer=True, confidence=0.9, answer_type="requires_context")
        agent = SupportAgent(llm=_mock_llm(assessment, _support_output()))
        result = await agent.generate("error 0x1A2B", [_ranked_doc()])
        assert result.retrieval_used is True

    @pytest.mark.asyncio
    async def test_falls_back_to_direct_when_no_documents_even_if_retrieval_needed(self):
        assessment = _assessment(can_answer=False, confidence=0.3, answer_type="requires_context")
        agent = SupportAgent(llm=_mock_llm(assessment, _support_output()))
        result = await agent.generate("complex issue", [])  # empty documents
        # Cannot do RAG without documents — must fall back to direct
        assert result.retrieval_used is False

    @pytest.mark.asyncio
    async def test_confidence_from_assessment_is_preserved_in_output(self):
        assessment = _assessment(confidence=0.77)
        agent = SupportAgent(llm=_mock_llm(assessment, _support_output()))
        result = await agent.generate("query", [])
        assert result.confidence == pytest.approx(0.77)

    @pytest.mark.asyncio
    async def test_citation_scores_overridden_on_rag_path(self):
        doc = _ranked_doc(doc_id="kb-001", rerank_score=0.95)
        assessment = _assessment(can_answer=False, confidence=0.3, answer_type="requires_context")
        citation = CitationOutput(
            document_id="kb-001", source="kb.txt",
            excerpt="Reset via Settings > Security.", rerank_score=0.0,
        )
        output = _support_output(citations=[citation])
        agent = SupportAgent(llm=_mock_llm(assessment, output))
        result = await agent.generate("query", [doc])
        assert result.citations[0].rerank_score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_raises_value_error_for_blank_query(self):
        agent = SupportAgent(llm=_mock_llm(_assessment(), _support_output()))
        with pytest.raises(ValueError, match="blank"):
            await agent.generate("   ", [])

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_assessment_failure(self):
        confidence_chain = AsyncMock()
        confidence_chain.ainvoke.side_effect = RuntimeError("Ollama offline")
        generate_chain = AsyncMock()
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        with pytest.raises(LLMError, match="Confidence assessment failed"):
            await agent.generate("query", [])

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_wrong_assessment_type(self):
        confidence_chain = AsyncMock()
        confidence_chain.ainvoke.return_value = {"can_answer_directly": True}  # dict, not model
        generate_chain = AsyncMock()
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        with pytest.raises(LLMError, match="Expected ConfidenceAssessment"):
            await agent.generate("query", [])

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_generation_failure(self):
        confidence_chain = AsyncMock()
        confidence_chain.ainvoke.return_value = _assessment()
        generate_chain = AsyncMock()
        generate_chain.ainvoke.side_effect = RuntimeError("generation failed")
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        with pytest.raises(LLMError, match="Direct support generation failed"):
            await agent.generate("query", [])

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_wrong_generation_type(self):
        confidence_chain = AsyncMock()
        confidence_chain.ainvoke.return_value = _assessment()
        generate_chain = AsyncMock()
        generate_chain.ainvoke.return_value = "plain string"
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        with pytest.raises(LLMError, match="Expected SupportOutput"):
            await agent.generate("query", [])


# ── SupportAgent triage ────────────────────────────────────────────────────────


class TestSupportAgentTriage:
    @pytest.mark.asyncio
    async def test_triage_returns_confidence_assessment(self):
        expected = _assessment()
        confidence_chain = AsyncMock()
        confidence_chain.ainvoke.return_value = expected
        generate_chain = AsyncMock()
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        result = await agent._assess_confidence("reset password")
        assert isinstance(result, ConfidenceAssessment)
        assert result.confidence == pytest.approx(expected.confidence)

    @pytest.mark.asyncio
    async def test_triage_sends_query_to_confidence_chain(self):
        expected = _assessment()
        confidence_chain = AsyncMock()
        confidence_chain.ainvoke.return_value = expected
        generate_chain = AsyncMock()
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        await agent._assess_confidence("my specific query")
        messages_arg = confidence_chain.ainvoke.call_args[0][0]
        human_content = messages_arg[1].content
        assert "my specific query" in human_content


# ── SupportAgent direct generation ────────────────────────────────────────────


class TestSupportAgentDirect:
    @pytest.mark.asyncio
    async def test_direct_returns_support_output(self):
        expected = _support_output()
        confidence_chain = AsyncMock()
        generate_chain = AsyncMock()
        generate_chain.ainvoke.return_value = expected
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        result = await agent._generate_direct("how to reset?")
        assert isinstance(result, SupportOutput)

    @pytest.mark.asyncio
    async def test_direct_prompt_contains_query(self):
        expected = _support_output()
        confidence_chain = AsyncMock()
        generate_chain = AsyncMock()
        generate_chain.ainvoke.return_value = expected
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        await agent._generate_direct("unique-query-string")
        messages_arg = generate_chain.ainvoke.call_args[0][0]
        human_content = messages_arg[1].content
        assert "unique-query-string" in human_content


# ── SupportAgent context generation ───────────────────────────────────────────


class TestSupportAgentContext:
    @pytest.mark.asyncio
    async def test_context_returns_support_output(self):
        expected = _support_output()
        confidence_chain = AsyncMock()
        generate_chain = AsyncMock()
        generate_chain.ainvoke.return_value = expected
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        result = await agent._generate_with_context("query", [_ranked_doc()])
        assert isinstance(result, SupportOutput)

    @pytest.mark.asyncio
    async def test_context_prompt_contains_document_id(self):
        expected = _support_output()
        confidence_chain = AsyncMock()
        generate_chain = AsyncMock()
        generate_chain.ainvoke.return_value = expected
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        doc = _ranked_doc(doc_id="special-doc-id")
        await agent._generate_with_context("query", [doc])
        messages_arg = generate_chain.ainvoke.call_args[0][0]
        full_text = " ".join(m.content for m in messages_arg)
        assert "special-doc-id" in full_text


# ── SupportAgent retry ────────────────────────────────────────────────────────


class TestSupportAgentRetry:
    @pytest.mark.asyncio
    async def test_retries_confidence_chain_on_transient_failure(self):
        expected_assessment = _assessment()
        confidence_chain = AsyncMock()
        confidence_chain.ainvoke.side_effect = [
            RuntimeError("timeout"),
            expected_assessment,
        ]
        generate_chain = AsyncMock()
        generate_chain.ainvoke.return_value = _support_output()
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        result = await agent.generate("query", [])
        assert isinstance(result, SupportOutput)
        assert confidence_chain.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_confidence_retries_exhausted(self):
        confidence_chain = AsyncMock()
        confidence_chain.ainvoke.side_effect = RuntimeError("always fails")
        generate_chain = AsyncMock()
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        with pytest.raises(LLMError):
            await agent.generate("query", [])
        assert confidence_chain.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_generate_chain_on_transient_failure(self):
        confidence_chain = AsyncMock()
        confidence_chain.ainvoke.return_value = _assessment()
        generate_chain = AsyncMock()
        generate_chain.ainvoke.side_effect = [
            RuntimeError("timeout"),
            _support_output(),
        ]
        llm = MagicMock()
        llm.with_structured_output.side_effect = [confidence_chain, generate_chain]
        agent = SupportAgent(llm=llm)
        result = await agent.generate("query", [])
        assert isinstance(result, SupportOutput)
        assert generate_chain.ainvoke.call_count == 2


# ── support_node ──────────────────────────────────────────────────────────────


class TestSupportNode:
    @pytest.mark.asyncio
    async def test_sets_current_node(self):
        update = await support_node(_state())
        assert update["current_node"] == "support"

    @pytest.mark.asyncio
    async def test_increments_step_count_from_zero(self):
        update = await support_node(_state())
        assert update["step_count"] == 1

    @pytest.mark.asyncio
    async def test_increments_step_count_from_existing(self):
        s = _state()
        s["step_count"] = 5
        update = await support_node(s)
        assert update["step_count"] == 6

    @pytest.mark.asyncio
    async def test_no_errors_key(self):
        update = await support_node(_state())
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_does_not_modify_query(self):
        update = await support_node(_state())
        assert "query" not in update


# ── generator_node route dispatch ────────────────────────────────────────────


class TestGeneratorNodeRouting:
    @pytest.mark.asyncio
    async def test_uses_support_agent_for_support_route(self):
        output = _support_output()
        mock_svc = AsyncMock()
        mock_svc.generate.return_value = output
        with patch("app.graph.nodes.generator.SupportAgent", return_value=mock_svc):
            update = await generator_node(_state_with_route("support"))
        mock_svc.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_research_agent_for_research_route(self):
        from app.agents.research_agent import ResearchOutput
        output = ResearchOutput(
            summary="a" * 10, answer="b" * 20, citations=[]
        )
        mock_svc = AsyncMock()
        mock_svc.generate.return_value = output
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_svc):
            update = await generator_node(_state_with_route("research"))
        mock_svc.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_research_agent_when_route_absent(self):
        from app.agents.research_agent import ResearchOutput
        output = ResearchOutput(summary="a" * 10, answer="b" * 20, citations=[])
        mock_svc = AsyncMock()
        mock_svc.generate.return_value = output
        state = _state_with_docs()
        # no route key set
        with patch("app.graph.nodes.generator.ResearchAgent", return_value=mock_svc):
            update = await generator_node(state)
        mock_svc.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_support_draft_is_valid_research_output_json(self):
        """SupportOutput serialises to JSON parseable by ResearchOutput."""
        output = _support_output(retrieval_used=True, confidence=0.88)
        mock_svc = AsyncMock()
        mock_svc.generate.return_value = output
        with patch("app.graph.nodes.generator.SupportAgent", return_value=mock_svc):
            update = await generator_node(_state_with_route("support"))
        from app.agents.research_agent import ResearchOutput
        parsed = ResearchOutput.model_validate_json(update["draft_response"])
        assert parsed.summary == output.summary

    @pytest.mark.asyncio
    async def test_support_route_records_error_on_failure(self):
        mock_svc = AsyncMock()
        mock_svc.generate.side_effect = LLMError("support agent crashed")
        with patch("app.graph.nodes.generator.SupportAgent", return_value=mock_svc):
            update = await generator_node(_state_with_route("support"))
        assert len(update["errors"]) == 1
        assert update["errors"][0]["node"] == "generator"
