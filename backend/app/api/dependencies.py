"""
FastAPI dependency providers for the workflow and approval service.

A single MemorySaver-backed compiled graph is created on first use and reused
for the process lifetime.  In production, replace the checkpointer with
AsyncPostgresSaver from PostgresCheckpointStore after calling its setup().

Call init_workflow() once during application startup (lifespan) to ensure
the graph is ready before any request arrives.
"""

import asyncio
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from app.core.logging import get_logger
from app.graph.workflow import compile_workflow
from app.services.approval_service import ApprovalService

_logger = get_logger(__name__)

_workflow: CompiledStateGraph | None = None
_approval_service: ApprovalService | None = None

# Running workflow tasks tracked to detect errors in background jobs.
_running_tasks: dict[str, asyncio.Task] = {}


def init_workflow(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Compile and cache the workflow.  Safe to call multiple times."""
    global _workflow, _approval_service
    cp = checkpointer if checkpointer is not None else MemorySaver()
    _workflow = compile_workflow(checkpointer=cp)
    _approval_service = ApprovalService(workflow=_workflow)
    _logger.info("workflow_initialised", checkpointer=type(cp).__name__)
    return _workflow


def get_workflow() -> CompiledStateGraph:
    """FastAPI dependency — returns the compiled workflow."""
    if _workflow is None:
        return init_workflow()
    return _workflow


def get_approval_service() -> ApprovalService:
    """FastAPI dependency — returns the shared ApprovalService."""
    if _approval_service is None:
        init_workflow()
    assert _approval_service is not None
    return _approval_service


def track_task(session_id: str, task: asyncio.Task) -> None:
    """Register a background workflow task so errors can be inspected."""
    _running_tasks[session_id] = task
    task.add_done_callback(lambda t: _running_tasks.pop(session_id, None))


def get_task(session_id: str) -> asyncio.Task | None:
    return _running_tasks.get(session_id)
