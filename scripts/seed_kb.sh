#!/bin/sh
# Seeds the ChromaDB knowledge base with sample documents for both agent
# collections. Runs once at stack startup via the `seed` Docker Compose
# service. Re-running is safe — chunk IDs are deterministic so existing
# entries are updated, not duplicated.
set -e

BASE="http://backend:8000/api/v1/ingest"

echo "==> Seeding research collection..."
curl -sf -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "research",
    "documents": [
      {
        "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains across multiple steps of computation in a cyclic manner.",
        "source": "langgraph-docs",
        "metadata": {"topic": "AI frameworks"}
      },
      {
        "content": "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation. A retriever fetches relevant documents; the generator conditions its output on those documents, reducing hallucination and improving factual accuracy.",
        "source": "rag-overview",
        "metadata": {"topic": "AI techniques"}
      },
      {
        "content": "CrossEncoder models take a query and a document as input and output a relevance score. Unlike bi-encoders, they compare the full pair together, producing more accurate scores at the cost of higher latency. BAAI/bge-reranker-large is a popular open-source CrossEncoder.",
        "source": "reranking-guide",
        "metadata": {"topic": "information retrieval"}
      }
    ]
  }'
echo " done."

echo "==> Seeding support collection..."
curl -sf -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "support",
    "documents": [
      {
        "content": "To reset your password: go to Settings then Account then Security then Change Password. Enter your current password, then your new password twice. Changes take effect immediately.",
        "source": "support-password-reset",
        "metadata": {"topic": "account management"}
      },
      {
        "content": "Human-in-the-loop (HITL) workflows pause automated processes to allow a human reviewer to approve, reject, or correct AI outputs before they are acted upon. This is critical in high-stakes domains.",
        "source": "hitl-overview",
        "metadata": {"topic": "AI safety"}
      }
    ]
  }'
echo " done."

echo "==> Knowledge base seed complete."
