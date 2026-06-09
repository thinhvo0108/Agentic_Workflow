from app.core.config import get_settings
from app.graph.state import ApprovalStatus


class ApprovalService:
    """Manages human-approval lifecycle: creation, polling, and resolution."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def create_approval_request(self, session_id: str) -> None:
        raise NotImplementedError

    async def get_approval_status(self, session_id: str) -> ApprovalStatus:
        raise NotImplementedError

    async def submit_decision(
        self,
        session_id: str,
        action: ApprovalStatus,
        reviewer_id: str,
        comment: str | None = None,
    ) -> None:
        raise NotImplementedError
