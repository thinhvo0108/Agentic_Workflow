from fastapi import APIRouter, HTTPException, status

from app.schemas.requests import ApprovalRequest, WorkflowRequest
from app.schemas.responses import (
    ApprovalResponse,
    ErrorResponse,
    WorkflowResponse,
    WorkflowStatusResponse,
)

router = APIRouter(prefix="/api/v1", tags=["workflow"])


@router.post(
    "/workflow",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new workflow request",
)
async def submit_workflow(request: WorkflowRequest) -> WorkflowStatusResponse:
    raise NotImplementedError


@router.get(
    "/workflow/{session_id}",
    response_model=WorkflowStatusResponse,
    summary="Get the status of an in-progress workflow",
)
async def get_workflow_status(session_id: str) -> WorkflowStatusResponse:
    raise NotImplementedError


@router.get(
    "/workflow/{session_id}/result",
    response_model=WorkflowResponse,
    summary="Retrieve the final approved result",
)
async def get_workflow_result(session_id: str) -> WorkflowResponse:
    raise NotImplementedError


@router.post(
    "/workflow/{session_id}/approve",
    response_model=ApprovalResponse,
    summary="Submit a human approval decision",
)
async def submit_approval(session_id: str, request: ApprovalRequest) -> ApprovalResponse:
    raise NotImplementedError
