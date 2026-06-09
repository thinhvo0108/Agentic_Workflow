# Agentic Workflow

A production-style AI workflow that routes user queries to specialized agents, retrieves and reranks documents, generates structured answers, evaluates groundedness, and gates every response behind a human approval step.

Built to demonstrate the engineering patterns used in enterprise AI systems: multi-agent orchestration, retrieval-augmented generation, LLM-as-judge evaluation, checkpointing, and human-in-the-loop control.

---

## How It Works

Every query travels through a fixed pipeline managed by LangGraph. Each box below is a node in the state machine.

```
User Query
    │
    ▼
┌─────────┐
│  Router │  LLM classifies intent → "research" or "support"
└────┬────┘
     │
  ┌──┴──┐
  │     │
  ▼     ▼
Research  Support
Agent    Agent
  │     │
  └──┬──┘
     │
     ▼
┌───────────┐
│ Retriever │  Top-10 chunks from ChromaDB (cosine similarity)
└─────┬─────┘
      │
      ▼
┌──────────┐
│ Reranker │  CrossEncoder scores each chunk; keeps top 3
└────┬─────┘
     │
     ▼
┌───────────┐
│ Generator │  LLM writes answer + inline citations from top-3 docs
└─────┬─────┘
      │
      ▼
┌──────────────────┐
│ Structured Output│  Pydantic-validates the draft JSON
└────────┬─────────┘
         │
         ▼
┌──────────────┐
│ Groundedness │  LLM extracts claims, labels each supported/unsupported
└──────┬───────┘
       │
       ▼
┌────────────┐
│ Checkpoint │  Persists full state to PostgreSQL
└─────┬──────┘
      │
      ▼ (workflow pauses here)
┌────────────────┐
│ Human Approval │  Reviewer calls POST /approve → approved or rejected
└───────┬────────┘
        │
    approved
        │
        ▼
┌────────────────┐
│ Final Response │  Assembles confidence scores + groundedness into response
└────────────────┘
```

### Node-by-Node Breakdown

| Node | What happens |
|---|---|
| **Router** | `ChatOllama` with structured output classifies the query. Returns `route` ("research"/"support") and a `router_confidence` score (LLM self-reported, 0–1). |
| **Research / Support Agent** | Specialized system prompts; research always retrieves, support retrieves when confidence is low. Both produce a `draft_response`. |
| **Retriever** | Embeds the query with `nomic-embed-text`, queries ChromaDB for the top-10 nearest chunks. Computes `retrieval_confidence` as a position-weighted mean of cosine similarities (rank-1 document weighted 2× rank-2, etc.). |
| **Reranker** | `BAAI/bge-reranker-large` CrossEncoder scores every (query, chunk) pair and keeps the top 3. Scores are sigmoid-normalised to [0, 1]. |
| **Generator** | `ChatOllama` with `with_structured_output(ResearchOutput)` writes a summary, detailed answer, and citations from the top-3 chunks. Citation scores are overridden with authoritative reranker values. |
| **Structured Output** | Re-validates the draft JSON as a `StructuredOutput` TypedDict. Computes `answer_confidence` as the mean rerank score of context documents. |
| **Groundedness** | A second LLM call extracts every factual claim from the answer, then labels each one supported/unsupported by the source documents. `groundedness_score = supported_count / total_count`. Non-blocking: a failure appends to errors and lets the workflow continue. |
| **Checkpoint** | Persists the full `AppState` to PostgreSQL via `AsyncPostgresSaver`. Every field is stored as plain JSON, enabling full state reconstruction. |
| **Human Approval** | LangGraph pauses (`interrupt_before`) and waits for a `POST /api/v1/workflow/{id}/approve` call. The graph resumes with the reviewer decision injected into state. |
| **Final Response** | Assembles the `FinalResponse` with confidence scores (`router * 0.2 + retrieval * 0.3 + answer * 0.5`) and the groundedness evaluation. |

---

## Tech Stack

### AI / ML

| Component | Technology |
|---|---|
| LLM | [Ollama](https://ollama.com) — `qwen3:14b` (default), `llama3.3:70b` (optional) |
| Embeddings | `nomic-embed-text` via Ollama |
| Reranker | `BAAI/bge-reranker-large` (CrossEncoder via `sentence-transformers`) |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` with PostgreSQL checkpointing |
| Agent framework | [LangChain](https://github.com/langchain-ai/langchain) + `ChatOllama` |

### Backend

| Component | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Vector store | ChromaDB (cosine distance) |
| Persistence | PostgreSQL 16 + asyncpg + LangGraph `AsyncPostgresSaver` |
| Observability | OpenTelemetry (OTLP/gRPC traces) + Prometheus metrics |
| Logging | structlog (JSON, with OTel trace/span IDs injected) |
| Retries | tenacity (exponential backoff, 3 attempts) |

### Frontend

| Component | Technology |
|---|---|
| Framework | React 18 + TypeScript |
| Build tool | Vite |
| UI library | Chakra UI v2 |
| Routing | React Router v6 |
| API polling | Custom `useWorkflowPoller` hook (1.5 s interval, stops at terminal states) |

### Infrastructure

| Component | Technology |
|---|---|
| Containerisation | Docker Compose (5 services) |
| CI/CD | GitHub Actions (lint → test → build → push to GHCR) |

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) ≥ 4.x (or Docker Engine + Compose v2)
- [Ollama](https://ollama.com) — only needed for local-without-Docker runs

For the Docker path everything else (Python, Node, Postgres, Chroma) runs inside containers.

---

## Quickstart (Docker Compose)

### 1. Clone and configure

```bash
git clone <repo-url>
cd agentic-workflow

cp backend/.env.example backend/.env   # edit if needed; defaults work out of the box
```

Default `.env` values that work without changes:

```env
POSTGRES_HOST=localhost
POSTGRES_DB=agentic_workflow
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
CHROMA_HOST=localhost
CHROMA_PORT=8001
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=qwen3:14b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### 2. Start all services

```bash
docker compose up --build
```

First run downloads images and builds containers (~5 min). Subsequent starts take ~20 s.

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| ChromaDB | http://localhost:8001 |
| Prometheus metrics | http://localhost:8000/metrics |

### 3. Pull the LLM models

In a separate terminal (models persist in the `ollama_data` Docker volume):

```bash
# Required
docker compose exec ollama ollama pull qwen3:14b
docker compose exec ollama ollama pull nomic-embed-text

# Optional — heavier but more capable
docker compose exec ollama ollama pull llama3.3:70b
```

`qwen3:14b` is ~9 GB. `nomic-embed-text` is ~274 MB.

### 4. Seed the knowledge base

The workflow retrieves from ChromaDB, so you need documents loaded before queries will return meaningful results. Use the ingestion API:

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains (or agents) across multiple steps of computation in a cyclic manner.",
        "source": "langgraph-docs",
        "metadata": {"topic": "AI frameworks", "type": "documentation"}
      },
      {
        "content": "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation. A retriever fetches relevant documents from a corpus; the generator then conditions its output on those documents, reducing hallucination and improving factual accuracy.",
        "source": "rag-overview",
        "metadata": {"topic": "AI techniques", "type": "research"}
      },
      {
        "content": "ChromaDB is an open-source embedding database. It stores vector embeddings alongside metadata and supports cosine similarity search. Embeddings are typically produced by a sentence-transformer model.",
        "source": "chromadb-docs",
        "metadata": {"topic": "vector databases", "type": "documentation"}
      },
      {
        "content": "Human-in-the-loop (HITL) workflows pause automated processes to allow a human reviewer to approve, reject, or correct AI outputs before they are acted upon. This is critical in high-stakes domains such as healthcare, legal, and finance.",
        "source": "hitl-overview",
        "metadata": {"topic": "AI safety", "type": "research"}
      },
      {
        "content": "CrossEncoder models take a query and a document as input and output a relevance score. Unlike bi-encoders, they compare the full pair together, producing more accurate scores at the cost of higher latency. BAAI/bge-reranker-large is a popular open-source CrossEncoder.",
        "source": "reranking-guide",
        "metadata": {"topic": "information retrieval", "type": "research"}
      }
    ]
  }'
```

Re-ingesting the same source is safe — the pipeline uses deterministic SHA-256 chunk IDs so duplicates are updated rather than re-inserted.

---

## Test Queries

These queries exercise different parts of the system. Send them through the UI at http://localhost:5173 or via `curl`.

### Research queries (routed to Research Agent)

```
What is LangGraph and how does it differ from standard LangChain?
How does retrieval-augmented generation reduce hallucination?
Explain how CrossEncoder reranking improves retrieval quality.
What are the trade-offs between bi-encoders and cross-encoders?
How does ChromaDB store and retrieve vector embeddings?
```

### Support / FAQ queries (routed to Support Agent)

```
How do I install ChromaDB?
What models does Ollama support?
Why is human-in-the-loop important in production AI systems?
What is the difference between approved and rejected workflow status?
How do I re-ingest a document without creating duplicates?
```

### Edge cases worth testing

```
# Short / ambiguous — tests router fallback
AI

# No relevant documents — groundedness should show 0 supported claims
What is the capital of France?

# Mixed research + support phrasing — tests routing boundary
Can you explain and help me troubleshoot RAG pipelines?
```

---

## API Reference

All endpoints are documented interactively at http://localhost:8000/docs.

### Submit a query

```bash
curl -X POST http://localhost:8000/api/v1/workflow \
  -H "Content-Type: application/json" \
  -d '{"query": "How does reranking improve retrieval quality?"}'
```

Returns `202 Accepted` with a `session_id`.

### Poll for status

```bash
curl http://localhost:8000/api/v1/workflow/{session_id}
```

Status values: `running` → `awaiting_approval` → `completed` / `rejected` / `failed`

### Approve or reject

```bash
curl -X POST http://localhost:8000/api/v1/workflow/{session_id}/approve \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "{session_id}",
    "action": "approved",
    "reviewer_id": "reviewer@example.com",
    "comment": "Answer looks accurate and well-cited."
  }'
```

### Get the final result

```bash
curl http://localhost:8000/api/v1/workflow/{session_id}/result
```

Response shape:

```json
{
  "session_id": "...",
  "summary": "...",
  "answer": "...",
  "citations": [
    {
      "document_id": "...",
      "source": "reranking-guide",
      "excerpt": "...",
      "relevance_score": 0.91
    }
  ],
  "route": "research",
  "approval_status": "approved",
  "confidence": {
    "router": 0.95,
    "retrieval": 0.78,
    "answer": 0.88,
    "overall": 0.86
  },
  "groundedness": {
    "groundedness_score": 0.8333,
    "supported_claims": [...],
    "unsupported_claims": [...],
    "evaluated_at": "2025-06-09T12:00:00Z"
  }
}
```

### Ingest documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"content": "...", "source": "my-source", "metadata": {}}]}'
```

---

## Confidence & Groundedness Scores

Every completed workflow returns two independent quality signals.

### Confidence scores

Computed deterministically from model outputs — no extra LLM call.

| Signal | How it is computed | Weight |
|---|---|---|
| `router` | LLM self-reported probability for the routing decision | 20% |
| `retrieval` | Position-weighted mean of ChromaDB cosine similarities (rank 1 = 1.0×, rank 2 = 0.5×, rank n = 1/n×) | 30% |
| `answer` | Mean CrossEncoder rerank score of the top-3 context documents | 50% |
| `overall` | Weighted sum of the three signals above | — |

All values are clamped to [0.0, 1.0] and rounded to 4 decimal places.

### Groundedness score

Requires a second LLM call (after answer generation).

The evaluator extracts every factual claim from the answer, then labels each one as supported or unsupported by the source documents:

```
groundedness_score = supported_claim_count / total_claim_count
```

This approach is more reliable than asking the LLM to output a float directly — the hard work is binary classification per claim; the numeric aggregation is deterministic.

A score near 1.0 means the answer is tightly grounded in retrieved sources. Near 0.0 means the model likely hallucinated.

---

## Running Tests

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Unit tests only (no infrastructure required)
pytest tests/unit -m "not integration" --override-ini="addopts="

# All tests with coverage
pytest
```

Test coverage by module:

| Module | What is tested |
|---|---|
| `test_confidence.py` | `_clamp`, `score_retrieval` (position weighting), `score_answer`, `score_overall` (weight contract) |
| `test_groundedness.py` | `build_groundedness_result` (score math, partitioning), service delegation, node state transitions |
| `test_research_agent.py` | Research agent generation, citation score override |
| `test_support_agent.py` | Support agent routing logic |
| `test_human_approval.py` | Approval state machine transitions |
| `test_observability.py` | OTel span creation, Prometheus metric recording |
| `test_checkpoints.py` (integration) | PostgreSQL checkpoint round-trip |

---

## Local Development (without Docker)

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Start PostgreSQL and ChromaDB separately (or use Docker for just those):
docker compose up postgres chromadb -d

# Run the API
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Make sure `VITE_API_PROXY_TARGET` is unset (or set to `http://localhost:8000`) so the dev proxy points at your local backend.

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── agents/          # router.py, research_agent.py, support_agent.py
│   │   ├── api/             # routes.py, dependencies.py
│   │   ├── checkpoints/     # PostgreSQL checkpoint models + repository
│   │   ├── core/            # config.py, logging.py, exceptions.py
│   │   ├── evaluation/      # groundedness evaluator, schemas, service
│   │   ├── graph/
│   │   │   ├── nodes/       # one file per workflow node
│   │   │   ├── state.py     # AppState TypedDict — single source of truth
│   │   │   └── workflow.py  # StateGraph definition + compile_workflow()
│   │   ├── observability/   # OTel tracing, Prometheus metrics, middleware
│   │   ├── rag/             # embeddings, vector_store, retriever, reranker, ingestion
│   │   ├── schemas/         # Pydantic request/response models
│   │   └── services/        # approval_service.py, confidence.py
│   └── tests/
│       ├── unit/
│       └── integration/
├── frontend/
│   └── src/
│       ├── api/             # typed fetch wrappers
│       ├── components/      # QueryForm, WorkflowStepper, DocumentsPanel, etc.
│       ├── hooks/           # useWorkflowPoller
│       ├── pages/           # HomePage, WorkflowPage
│       └── types/           # TypeScript interfaces mirroring API schemas
├── .github/workflows/
│   ├── backend.yml          # ruff lint → mypy → pytest (unit only)
│   ├── frontend.yml         # tsc --noEmit → vite build
│   └── docker.yml           # build backend + frontend images → push to GHCR
└── docker-compose.yml
```

---

## CI/CD

Three GitHub Actions pipelines run on every push:

| Pipeline | Triggers on | Jobs |
|---|---|---|
| `backend.yml` | `backend/**` changes | ruff lint → mypy type-check → pytest (unit, with Codecov) |
| `frontend.yml` | `frontend/**` changes | tsc type-check → vite build (artifact uploaded) |
| `docker.yml` | any push | build both Docker images; push to GHCR only on `main` |

Docker images are cached via GitHub Actions layer cache (`type=gha`) so incremental builds are fast.

---

## Configuration Reference

All settings are read from environment variables (or `backend/.env`).

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_DEFAULT_MODEL` | `qwen3:14b` | Model used for routing, generation, and evaluation |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_TIMEOUT` | `120` | Per-request timeout in seconds |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8001` | ChromaDB port (host-side; containers override to 8000 internally) |
| `CHROMA_COLLECTION_NAME` | `knowledge_base` | Collection name |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_DB` | `agentic_workflow` | Database name |
| `POSTGRES_USER` | `postgres` | Username |
| `POSTGRES_PASSWORD` | `postgres` | Password |
| `RAG_RETRIEVAL_TOP_K` | `10` | Chunks retrieved from ChromaDB |
| `RAG_RERANKER_TOP_N` | `3` | Chunks kept after reranking |
| `APPROVAL_TIMEOUT_SECONDS` | `3600` | How long the workflow waits for a human decision |

---

## GPU Support

Ollama can use an NVIDIA GPU for faster inference. Uncomment the `deploy` block in `docker-compose.yml` (requires `nvidia-container-toolkit` installed on the host):

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

On Apple Silicon, Ollama uses the Metal GPU automatically when run natively (outside Docker).
