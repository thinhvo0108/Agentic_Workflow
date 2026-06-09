from app.core.config import get_settings


class PostgresCheckpointStore:
    """LangGraph-compatible checkpoint backend using PostgreSQL via asyncpg."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._pool: object | None = None

    async def setup(self) -> None:
        """Initialize the connection pool and create the checkpoints table if absent."""
        raise NotImplementedError

    async def teardown(self) -> None:
        raise NotImplementedError

    def as_langgraph_checkpointer(self) -> object:
        """Return a langgraph_checkpoint_postgres.AsyncPostgresSaver instance."""
        raise NotImplementedError
