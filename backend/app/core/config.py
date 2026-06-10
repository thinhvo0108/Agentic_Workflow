from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")


class OllamaSettings(BaseSettings):
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2:latest"
    embedding_model: str = "nomic-embed-text"
    timeout: int = 120

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", env_file=".env", extra="ignore")


class ChromaSettings(BaseSettings):
    host: str = "localhost"
    port: int = 8001
    collection_name: str = "knowledge_base"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    model_config = SettingsConfigDict(env_prefix="CHROMA_", env_file=".env", extra="ignore")


class PostgresSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    db: str = "agentic_workflow"
    user: str = "postgres"
    password: str = "postgres"
    pool_size: int = 10
    max_overflow: int = 20

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")


class LangGraphSettings(BaseSettings):
    checkpoint_table: str = "checkpoints"
    max_steps: int = 20

    model_config = SettingsConfigDict(env_prefix="LANGGRAPH_", env_file=".env", extra="ignore")


class RAGSettings(BaseSettings):
    retrieval_top_k: int = Field(default=10, ge=1, le=100)
    reranker_top_n: int = Field(default=3, ge=1, le=20)
    reranker_model: str = "BAAI/bge-reranker-large"

    @model_validator(mode="after")
    def validate_top_n_lte_top_k(self) -> "RAGSettings":
        if self.reranker_top_n > self.retrieval_top_k:
            raise ValueError("reranker_top_n must be <= retrieval_top_k")
        return self

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ApprovalSettings(BaseSettings):
    timeout_seconds: int = Field(default=3600, ge=60)
    webhook_url: str = ""

    model_config = SettingsConfigDict(env_prefix="APPROVAL_", env_file=".env", extra="ignore")


class OtelSettings(BaseSettings):
    service_name: str = "agentic-workflow"
    exporter_otlp_endpoint: str = "http://localhost:4317"
    enabled: bool = False

    model_config = SettingsConfigDict(env_prefix="OTEL_", env_file=".env", extra="ignore")


class LangSmithSettings(BaseSettings):
    tracing_v2: bool = False
    api_key: str = ""
    project: str = "agentic-workflow"

    model_config = SettingsConfigDict(env_prefix="LANGCHAIN_", env_file=".env", extra="ignore")


class Settings(BaseSettings):
    app: AppSettings = Field(default_factory=AppSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    langgraph: LangGraphSettings = Field(default_factory=LangGraphSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    approval: ApprovalSettings = Field(default_factory=ApprovalSettings)
    otel: OtelSettings = Field(default_factory=OtelSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
