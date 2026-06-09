# CLAUDE.md

## Project Overview

This project demonstrates a production-style AI workflow built using LangGraph and local LLMs running on Ollama.

The system routes incoming requests to specialized agents, performs retrieval-augmented generation (RAG), reranks retrieved documents, generates structured outputs, supports checkpointing, and includes a human approval step before producing final responses.

The goal is to showcase practical AI engineering patterns commonly used in enterprise environments.

---

## Architecture

User Request
|
v
Routing Node
|
+---+---+
|       |
v       v
Research Support
Agent    Agent
|
v
Retrieve Documents
|
v
Rerank Results
|
v
Generate Draft
|
v
Structured Output
|
v
Checkpoint
|
v
Human Approval
|
v
Final Response

---

## Technical Stack

### LLM

* Ollama
* qwen3:14b
* llama3.3:70b (optional)

### Workflow

* LangGraph

### Agent Framework

* LangChain

### Vector Database

* ChromaDB

### Embeddings

* nomic-embed-text

### Reranking

* BAAI/bge-reranker-large
* sentence-transformers CrossEncoder

### API

* FastAPI

### Persistence

* PostgreSQL

### Frontend

* React
* TypeScript
* Vite

### Observability

* OpenTelemetry
* LangSmith (optional)

---

## Project Structure

backend/

```
app/

    graph/
        workflow.py
        state.py
        nodes/

    agents/
        router.py
        research_agent.py
        support_agent.py

    rag/
        retriever.py
        reranker.py
        embeddings.py

    schemas/
        requests.py
        responses.py

    checkpoints/
        postgres_checkpoint.py

    api/
        routes.py

    services/
        approval_service.py

    observability/
        tracing.py
```

frontend/

```
src/
    pages/
    components/
    api/
```

---

## Workflow Requirements

### Router Node

Responsibilities:

* Determine user intent
* Route to:

  * Research Agent
  * Support Agent

Output:

{
"route": "research"
}

or

{
"route": "support"
}

---

### Research Agent

Handles:

* Product research
* Technical questions
* Market analysis
* Knowledge exploration

Must always invoke retrieval pipeline.

---

### Support Agent

Handles:

* FAQ
* Customer support
* Troubleshooting

Must invoke retrieval when confidence is low.

---

### Retrieval Node

Retrieve top 10 chunks from vector database.

Output:

{
"documents": [...]
}

---

### Reranking Node

Use CrossEncoder reranker.

Input:

query + retrieved documents

Output:

top 3 most relevant documents

---

### Draft Generation Node

Generate response using:

* user query
* reranked documents

Must produce citations.

---

### Structured Output Node

Convert output into strict Pydantic schema.

Example:

{
"summary": "...",
"answer": "...",
"citations": [...]
}

---

### Checkpoint Node

Persist state after every major stage.

Store:

* query
* route
* retrieved docs
* reranked docs
* draft
* final output

Checkpoint backend:

PostgreSQL

---

### Human Approval Node

Status values:

* pending
* approved
* rejected

Workflow pauses until approval arrives.

---

### Final Response Node

Return approved answer.

---

## Engineering Standards

### Type Safety

* Pydantic v2
* Strict typing

### Testing

* pytest

Required tests:

* Router tests
* Retrieval tests
* Reranking tests
* Graph integration tests

### Logging

Structured JSON logging.

Never use print statements.

### Error Handling

All nodes must:

* catch exceptions
* update state
* log failures

### Security

Never execute arbitrary code from prompts.

Sanitize all user inputs.

---

## Success Criteria

The project should demonstrate:

* Multi-agent orchestration
* RAG pipeline
* Reranking
* Structured outputs
* Human-in-the-loop workflow
* Checkpointing
* Local LLM deployment
* Production architecture

This project is intended as a portfolio-quality demonstration for senior AI engineering interviews.
