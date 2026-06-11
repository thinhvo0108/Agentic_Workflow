# CLAUDE.md

## Project Overview

Production-style AI workflow built with LangGraph and local LLMs via Ollama. Routes queries to specialized agents, performs RAG with reranking, evaluates answer quality via groundedness checking and LLM-as-a-judge, auto-approves high-confidence responses, and falls back to human review for edge cases.

Portfolio-quality demonstration for senior AI engineering interviews.

---

## Architecture

Full node pipeline (left to right / top to bottom):

```
START → router → [research | support] → retriever → reranker
      → generator → structured_output → groundedness → llm_judge
      → checkpoint → auto_approval_gate
                          ├─ confidence ≥ 0.70 AND judge ≥ 0.70 → final_response → [knowledge_update | END]
                          └─ either fails → web_search → human_approval
                                                              ├─ approved → final_response → knowledge_update
                                                              └─ rejected → END
```

Key design decisions:
- **Autonomous first**: auto-approval gate bypasses human review when both confidence and judge score clear 70%.
- **Judge-as-veto**: LLM judge can block auto-approval even when retrieval confidence is high.
- **Feedback loop**: manually approved Q&A is ingested back into the agent's ChromaDB collection, improving future auto-approval rates.
- **Multi-tenancy**: each agent type has its own isolated ChromaDB collection (`knowledge_base_research`, `knowledge_base_support`).

---

## Technical Stack

### LLM

* Ollama — `llama3.2:latest` (default, pulled automatically by Docker Compose)
* Embedding model — `nomic-embed-text` (pulled automatically)

### Workflow

* LangGraph `StateGraph` with PostgreSQL checkpointing (`AsyncPostgresSaver`)
* `interrupt_before=["human_approval"]` for durable human-in-the-loop pause/resume

### Agent Framework

* LangChain + `ChatOllama` with `with_structured_output()`

### Vector Database

* ChromaDB — collection-per-agent-type, cosine distance

### Reranking

* `BAAI/bge-reranker-large` via `sentence-transformers` CrossEncoder

### API

* FastAPI + Uvicorn

### Persistence

* PostgreSQL 16 + asyncpg
  * LangGraph `AsyncPostgresSaver` — graph state / checkpoints
  * `CheckpointRepository` — custom audit records table

### Frontend

* React 18 + TypeScript + Vite + Chakra UI v2

### Observability

* OpenTelemetry (OTLP/gRPC traces) — every node wrapped via `observe_node()` in `workflow.py`
* Prometheus metrics — `/metrics` endpoint
* structlog — JSON structured logging with OTel trace/span IDs injected
* LangSmith (optional, disabled by default)

---

## Project Structure

```
backend/app/
    agents/
        router.py               # RouterAgent — ChatOllama with structured output
        research_agent.py       # ResearchAgent — always uses retrieved docs
        support_agent.py        # SupportAgent — two-pass triage + optional RAG
    api/
        routes.py               # POST /workflow, GET /workflow/{id}, POST /approve, POST /ingest
        dependencies.py         # FastAPI DI: workflow, approval_service, task tracking
    checkpoints/
        postgres_checkpoint.py  # LangGraph AsyncPostgresSaver wrapper
        repository.py           # CheckpointRepository — custom audit table
        models.py
    core/
        config.py               # Settings (ChromaSettings.collection_for, etc.)
        logging.py              # structlog JSON config
        exceptions.py
    evaluation/
        judge.py                # LLM-as-a-judge implementation
        judge_schemas.py        # Pydantic schemas for judge output
        evaluator.py            # Groundedness evaluator
        service.py
        schemas.py
    graph/
        state.py                # AppState TypedDict — single source of truth
        workflow.py             # StateGraph definition + compile_workflow()
        nodes/
            router.py           # Classifies query → "research" | "support"
            research.py         # Path marker, sets route
            support.py          # Path marker, sets route
            retriever.py        # ChromaDB top-K query, per-agent collection
            reranker.py         # CrossEncoder reranking, keeps top-N
            generator.py        # LLM draft generation with citations
            structured_output.py # Pydantic validation, answer_confidence
            groundedness.py     # Claim-level hallucination detection
            llm_judge.py        # 4-dimension quality scoring (auto_approve / needs_review)
            checkpoint.py       # Persists state to PostgreSQL
            auto_approval.py    # Gates on confidence AND judge score ≥ 0.70
            web_search.py       # DuckDuckGo results for human reviewer context
            human_approval.py   # Interrupt node — waits for POST /approve
            final_response.py   # Assembles FinalResponse, computes WorkflowMetrics
            knowledge_update.py # Ingests approved Q&A back into agent's collection
    observability/
        tracing.py              # OTel configure
        metrics.py              # Prometheus counters/histograms
        middleware.py           # Request tracing middleware
        node_telemetry.py       # observe_node() wrapper applied in workflow.py
        token_tracker.py        # Token usage tracking per node
    rag/
        embeddings.py           # EmbeddingService — nomic-embed-text via Ollama
        vector_store.py         # VectorStoreClient — ChromaDB per-collection
        retriever.py            # RetrieverService
        reranker.py             # RerankerService — BAAI/bge-reranker-large
        ingestion.py            # IngestionPipeline — chunk, embed, upsert
    schemas/
        requests.py             # WorkflowRequest, ApprovalRequest, IngestRequest
        responses.py            # WorkflowResponse, DraftResponse, JudgeResult, etc.
    services/
        approval_service.py     # State reads, submit_decision, resume graph
        confidence.py           # score_retrieval, score_answer, score_overall

scripts/
    seed_kb.sh                  # Seeds sample docs into research + support collections on startup

frontend/src/
    api/                        # Typed fetch wrappers
    components/
        WorkflowStepper.tsx     # All 13 pipeline steps including llm_judge
        ApprovalPanel.tsx       # Reviewer UI — shows which condition failed gate
        ConfidenceStats.tsx
        DocumentsPanel.tsx
        FinalResponsePanel.tsx
        WorkflowMetricsPanel.tsx
    hooks/
        useWorkflowPoller.ts    # 1.5s polling, stops at terminal states
    pages/
        HomePage.tsx
        WorkflowPage.tsx
    types/
        workflow.ts             # TypeScript interfaces mirroring API schemas
```

---

## Workflow Node Contracts

### auto_approval_gate
Requires **both**:
1. `score_overall(router_conf, retrieval_conf, answer_conf) >= 0.70`
2. `judge_result.overall_score >= 0.70` (or judge did not run — graceful degradation)

### llm_judge
Scores on 4 dimensions: faithfulness (40%), relevance (30%), completeness (20%), coherence (10%).
Outputs `recommendation`: `"auto_approve"` (≥ 0.70) or `"needs_review"` (< 0.70).

### groundedness
Extracts factual claims from the answer and labels each `supported`/`unsupported` against source docs.
`groundedness_score = supported / total`. Non-blocking — failure appends to `errors`, workflow continues.

### knowledge_update
Only runs after **manual** approval (not auto-approved). Stores `{query}\n\n{answer}` back into the same
agent collection that served the query. This improves future auto-approval rates for similar queries.

---

## State Design

`AppState` is a `TypedDict` (required by LangGraph).
- `errors` and `total_tokens` use `operator.add` reducer — nodes append/accumulate, never overwrite.
- All nested types are also `TypedDict` (not Pydantic) for JSON-serializable checkpoint compatibility.
- `initial_state()` in `state.py` is the canonical constructor.

---

## Engineering Standards

### Type Safety
* Pydantic v2 for API schemas and external-boundary validation
* TypedDict + strict typing for internal graph state

### Testing
* pytest — unit tests in `tests/unit/`, integration in `tests/integration/`
* Unit tests run without infrastructure (all external services mocked)
* Integration tests require a live PostgreSQL instance

### Logging
* structlog JSON — never use `print()`
* OTel trace_id / span_id injected automatically via `node_telemetry.py`

### Error Handling
* Every node catches exceptions, appends to `state["errors"]`, logs the failure
* Only the router node short-circuits to END on error; all others continue

### Security
* Never execute arbitrary code from prompts
* Sanitize all user inputs (`_UNSAFE_PATTERN` in `requests.py`)
* CORS restricted to `[]` in non-development environments

---

## Environment

Default model set in `docker-compose.yml` (`OLLAMA_DEFAULT_MODEL: llama3.2:latest`) overrides
`backend/.env`. Both files must agree on `POSTGRES_PASSWORD` or the backend fails to start.

Knowledge base is seeded automatically via the `seed` Docker Compose service on every stack start
(idempotent — SHA-256 chunk IDs prevent duplicates).

---

## Success Criteria

* Multi-agent orchestration (router → research/support)
* RAG pipeline (embed → retrieve → rerank → generate)
* Groundedness checking (claim-level hallucination detection)
* LLM-as-a-judge evaluation (holistic quality scoring)
* Auto-approval gate (autonomous operation for high-confidence queries)
* Human-in-the-loop fallback (durable pause/resume via LangGraph interrupt)
* Knowledge base self-improvement (approved Q&A re-ingested per agent collection)
* Multi-tenancy (isolated ChromaDB collection per agent type)
* Checkpointing (full state persisted to PostgreSQL, crash-recoverable)
* Structured outputs (strict Pydantic schemas end-to-end)
* Observability (OTel traces + Prometheus metrics + structured logs)
* Local LLM deployment (Ollama, no external API required)
* Production architecture (FastAPI, Docker Compose, CI/CD via GitHub Actions)
