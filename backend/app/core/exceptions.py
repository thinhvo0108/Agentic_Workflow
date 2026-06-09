from typing import Any


class AgenticWorkflowError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AgenticWorkflowError):
    """Raised when application configuration is invalid."""


class WorkflowError(AgenticWorkflowError):
    """Raised when the LangGraph workflow encounters an unrecoverable error."""


class NodeError(WorkflowError):
    """Raised when a specific workflow node fails."""

    def __init__(self, node_name: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"Node '{node_name}' failed: {message}", details)
        self.node_name = node_name


class RetrievalError(AgenticWorkflowError):
    """Raised when document retrieval from ChromaDB fails."""


class RerankingError(AgenticWorkflowError):
    """Raised when the reranking step fails."""


class EmbeddingError(AgenticWorkflowError):
    """Raised when embedding generation fails."""


class CheckpointError(AgenticWorkflowError):
    """Raised when persisting or loading a checkpoint fails."""


class ApprovalError(AgenticWorkflowError):
    """Raised when the human approval step encounters an error."""


class ApprovalTimeoutError(ApprovalError):
    """Raised when the approval window expires."""


class ValidationError(AgenticWorkflowError):
    """Raised when input or output validation fails."""


class LLMError(AgenticWorkflowError):
    """Raised when the LLM call fails or returns an unexpected response."""
