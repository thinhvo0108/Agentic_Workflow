"""
Workflow API routes.

Endpoints
---------
POST /api/v1/workflow
    Submit a query.  The graph runs in the background and pauses at
    human_approval.  Returns 202 with session_id immediately.

GET /api/v1/workflow/{session_id}
    Poll for current status: running | awaiting_approval | completed |
    rejected | failed | not_found.

GET /api/v1/workflow/{session_id}/result
    Retrieve the approved final response.  404 if not yet completed.

POST /api/v1/workflow/{session_id}/approve
    Submit an approval decision (approved | rejected).
    The graph resumes synchronously; when this endpoint returns the
    workflow has fully completed.
"""

import asyncio
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import (
    get_approval_service,
    get_workflow,
    track_task,
)
from app.core.exceptions import ApprovalError, EmbeddingError, RetrievalError
from app.core.logging import get_logger
from app.graph.state import initial_state
from app.schemas.requests import ApprovalRequest, IngestRequest, WorkflowRequest
from app.schemas.responses import (
    ApprovalResponse,
    Citation,
    ConfidenceScores,
    DraftResponse,
    EvaluatedClaim,
    GroundednessResult,
    WorkflowResponse,
    WorkflowStatusResponse,
)
from app.services.approval_service import ApprovalService

_logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["workflow"])


# ── Helper ─────────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(UTC)


# ── Submit workflow ────────────────────────────────────────────────────────────


@router.post(
    "/workflow",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new workflow request",
)
async def submit_workflow(
    request: WorkflowRequest,
    approval_svc: ApprovalService = Depends(get_approval_service),
    workflow=Depends(get_workflow),
) -> WorkflowStatusResponse:
    """Start the workflow asynchronously.

    The graph runs in the background and pauses before the human_approval node.
    Poll GET /workflow/{session_id} to track progress.
    """
    session_id = request.session_id or str(uuid.uuid4())
    state = initial_state(
        session_id=session_id,
        query=request.query,
        metadata=request.metadata,
    )
    config = {"configurable": {"thread_id": session_id}}

    async def _run():
        try:
            await workflow.ainvoke(state, config)
        except Exception as exc:
            _logger.error(
                "background_workflow_failed", session_id=session_id, error=str(exc)
            )

    task = asyncio.create_task(_run())
    track_task(session_id, task)

    _logger.info("workflow_submitted", session_id=session_id, query_len=len(request.query))
    return WorkflowStatusResponse(
        session_id=session_id,
        status="running",
        current_node=None,
        error=None,
        created_at=_now(),
        updated_at=_now(),
    )


# ── Get status ─────────────────────────────────────────────────────────────────


@router.get(
    "/workflow/{session_id}",
    response_model=WorkflowStatusResponse,
    summary="Get the current status of a workflow session",
)
async def get_workflow_status(
    session_id: str,
    approval_svc: ApprovalService = Depends(get_approval_service),
) -> WorkflowStatusResponse:
    """Poll for workflow progress.

    Status values
    -------------
    running            Graph is still executing nodes.
    awaiting_approval  Graph is paused pending a human decision.
    completed          Workflow approved and final response assembled.
    rejected           Reviewer rejected the answer.
    failed             An unrecoverable error occurred.
    not_found          No session with this ID exists.
    """
    wf_status = await approval_svc.get_status(session_id)
    current_node = await approval_svc.get_current_node(session_id)

    error: str | None = None
    if wf_status == "failed":
        state = await approval_svc.get_state(session_id)
        if state and state.get("errors"):
            error = state["errors"][-1].get("message")

    return WorkflowStatusResponse(
        session_id=session_id,
        status=wf_status,  # type: ignore[arg-type]
        current_node=current_node,
        error=error,
        created_at=_now(),
        updated_at=_now(),
    )


# ── Get result ─────────────────────────────────────────────────────────────────


@router.get(
    "/workflow/{session_id}/result",
    response_model=WorkflowResponse,
    summary="Retrieve the approved final response",
)
async def get_workflow_result(
    session_id: str,
    approval_svc: ApprovalService = Depends(get_approval_service),
) -> WorkflowResponse:
    """Return the approved answer.

    Only available when status = completed.  Returns 404 for any other status.
    """
    final = await approval_svc.get_final_response(session_id)
    if final is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No approved result for session '{session_id}'",
        )

    api_citations = [
        Citation(
            document_id=c.get("document_id", ""),
            source=c.get("source", ""),
            excerpt=c.get("excerpt", ""),
            relevance_score=float(c.get("rerank_score", 0.0)),
        )
        for c in (final.get("citations") or [])
    ]

    conf_data = final.get("confidence")
    api_confidence: ConfidenceScores | None = None
    if conf_data:
        api_confidence = ConfidenceScores(
            router=float(conf_data.get("router", 0.0)),
            retrieval=float(conf_data.get("retrieval", 0.0)),
            answer=float(conf_data.get("answer", 0.0)),
            overall=float(conf_data.get("overall", 0.0)),
        )

    gnd_data = final.get("groundedness")
    api_groundedness: GroundednessResult | None = None
    if gnd_data:
        api_groundedness = GroundednessResult(
            groundedness_score=float(gnd_data.get("groundedness_score", 0.0)),
            supported_claims=[
                EvaluatedClaim(
                    claim=c["claim"],
                    supported=c["supported"],
                    source_document_ids=c.get("source_document_ids", []),
                    reasoning=c["reasoning"],
                )
                for c in (gnd_data.get("supported_claims") or [])
            ],
            unsupported_claims=[
                EvaluatedClaim(
                    claim=c["claim"],
                    supported=c["supported"],
                    source_document_ids=c.get("source_document_ids", []),
                    reasoning=c["reasoning"],
                )
                for c in (gnd_data.get("unsupported_claims") or [])
            ],
            evaluated_at=gnd_data.get("evaluated_at", ""),
        )

    return WorkflowResponse(
        session_id=final["session_id"],
        summary=final["summary"],
        answer=final["answer"],
        citations=api_citations,
        route=final["route"],
        approval_status=final["approval_status"],
        confidence=api_confidence,
        groundedness=api_groundedness,
    )


# ── Get draft for approval review ─────────────────────────────────────────────


@router.get(
    "/workflow/{session_id}/draft",
    response_model=DraftResponse,
    summary="Retrieve the draft response pending human approval",
)
async def get_workflow_draft(
    session_id: str,
    approval_svc: ApprovalService = Depends(get_approval_service),
) -> DraftResponse:
    """Return the structured output available for the approver to review.

    Only available when status = awaiting_approval.  Returns 404 otherwise.
    """
    state = await approval_svc.get_state(session_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found")

    so = state.get("structured_output")
    if so is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No draft available for session '{session_id}'",
        )

    citations = [
        Citation(
            document_id=c.get("document_id", ""),
            source=c.get("source", ""),
            excerpt=c.get("excerpt", ""),
            relevance_score=float(c.get("rerank_score", 0.0)),
        )
        for c in (so.get("citations") or [])
    ]

    conf_data = state.get("confidence") or {}
    router_conf = float(state.get("router_confidence") or 0.0)
    retrieval_conf = float(state.get("retrieval_confidence") or 0.0)
    answer_conf = float(state.get("answer_confidence") or 0.0)
    from app.services.confidence import score_overall
    api_confidence = ConfidenceScores(
        router=router_conf,
        retrieval=retrieval_conf,
        answer=answer_conf,
        overall=score_overall(router_conf, retrieval_conf, answer_conf),
    ) if any([router_conf, retrieval_conf, answer_conf]) else None

    gnd_data = state.get("groundedness")
    api_groundedness: GroundednessResult | None = None
    if gnd_data:
        api_groundedness = GroundednessResult(
            groundedness_score=float(gnd_data.get("groundedness_score", 0.0)),
            supported_claims=[
                EvaluatedClaim(
                    claim=c["claim"],
                    supported=c["supported"],
                    source_document_ids=c.get("source_document_ids", []),
                    reasoning=c["reasoning"],
                )
                for c in (gnd_data.get("supported_claims") or [])
            ],
            unsupported_claims=[
                EvaluatedClaim(
                    claim=c["claim"],
                    supported=c["supported"],
                    source_document_ids=c.get("source_document_ids", []),
                    reasoning=c["reasoning"],
                )
                for c in (gnd_data.get("unsupported_claims") or [])
            ],
            evaluated_at=gnd_data.get("evaluated_at", ""),
        )

    return DraftResponse(
        session_id=session_id,
        query=state.get("query", ""),
        route=state.get("route") or "research",
        summary=so.get("summary", ""),
        answer=so.get("answer", ""),
        citations=citations,
        confidence=api_confidence,
        groundedness=api_groundedness,
    )


# ── Submit approval ────────────────────────────────────────────────────────────


@router.post(
    "/workflow/{session_id}/approve",
    response_model=ApprovalResponse,
    summary="Submit an approval decision (approved | rejected)",
)
async def submit_approval(
    session_id: str,
    request: ApprovalRequest,
    approval_svc: ApprovalService = Depends(get_approval_service),
) -> ApprovalResponse:
    """Inject a reviewer decision and resume the workflow.

    The graph runs to completion synchronously within this request.
    After this endpoint returns, the session status will be either
    'completed' (approved) or 'rejected'.
    """
    if session_id != request.session_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Path session_id does not match request body session_id",
        )

    try:
        await approval_svc.submit_decision(
            session_id=session_id,
            action=request.action,  # type: ignore[arg-type]
            reviewer_id=request.reviewer_id,
            comment=request.comment,
        )
    except ApprovalError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    _logger.info(
        "approval_api_complete",
        session_id=session_id,
        action=request.action,
        reviewer=request.reviewer_id,
    )
    return ApprovalResponse(
        session_id=session_id,
        action=request.action,  # type: ignore[arg-type]
        reviewer_id=request.reviewer_id,
        comment=request.comment,
    )


# ── Ingest documents ───────────────────────────────────────────────────────────


@router.post(
    "/ingest",
    status_code=status.HTTP_200_OK,
    summary="Ingest documents into the knowledge base",
)
async def ingest_documents(request: IngestRequest) -> dict:
    """Split, embed, and upsert documents into ChromaDB.

    Re-ingesting the same source is safe — chunk IDs are deterministic so
    existing entries are updated rather than duplicated.
    """
    from app.rag.ingestion import IngestionPipeline, IngestDocument

    docs: list[IngestDocument] = [
        IngestDocument(
            content=d.content,
            source=d.source,
            metadata=d.metadata,
        )
        for d in request.documents
    ]

    try:
        pipeline = IngestionPipeline()
        chunk_count = await pipeline.ingest(docs)
    except (EmbeddingError, RetrievalError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )

    _logger.info("ingest_complete", doc_count=len(docs), chunk_count=chunk_count)
    return {"documents_ingested": len(docs), "chunks_stored": chunk_count}
