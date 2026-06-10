"""
LangGraph workflow definition.

Each node is wrapped with observe_node() which adds:
  • An OTel span   (workflow.node.<name>)
  • Duration histogram   (workflow_node_duration_seconds)
  • Error counter        (workflow_node_errors_total)
  • trace_id / span_id injected into structlog context vars for the node body
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.state import AppState
from app.observability.node_telemetry import observe_node


def _route_decision(state: AppState) -> str:
    """Conditional edge after the router node.

    Routes to "research" or "support" on a clean run.
    Falls through to END if errors were recorded (router failed).
    """
    if state.get("errors"):
        return END
    route = state.get("route")
    return route if route in ("research", "support") else END


def _post_final_decision(state: AppState) -> str:
    """Conditional edge after final_response.

    Manual approval → knowledge_update (ingest approved Q&A into ChromaDB).
    Auto approval   → END (KB already has good coverage for high-confidence queries).
    """
    if not state.get("auto_approved", False):
        return "knowledge_update"
    return END


def _pre_approval_decision(state: AppState) -> str:
    """Conditional edge after auto_approval_gate.

    Routes to final_response when the gate auto-approved the response
    (overall confidence >= threshold).  Otherwise defers to human_approval.
    """
    if state.get("approval_status") == "approved":
        return "final_response"
    return "human_approval"


def _approval_decision(state: AppState) -> str:
    """Conditional edge after the human-approval node.

    "approved"  → proceed to final_response
    "rejected"  → terminate (END)
    "pending"   → loop back to human_approval (graph stays interrupted)
    """
    status = state.get("approval_status", "pending")
    if status == "approved":
        return "final_response"
    if status == "rejected":
        return END
    return "human_approval"


def build_workflow() -> StateGraph:
    """Construct the LangGraph state machine (not yet compiled)."""
    from app.graph.nodes.auto_approval import auto_approval_gate_node
    from app.graph.nodes.checkpoint import checkpoint_node
    from app.graph.nodes.final_response import final_response_node
    from app.graph.nodes.generator import generator_node
    from app.graph.nodes.groundedness import groundedness_node
    from app.graph.nodes.human_approval import human_approval_node
    from app.graph.nodes.knowledge_update import knowledge_update_node
    from app.graph.nodes.reranker import reranker_node
    from app.graph.nodes.research import research_node
    from app.graph.nodes.retriever import retriever_node
    from app.graph.nodes.router import router_node
    from app.graph.nodes.structured_output import structured_output_node
    from app.graph.nodes.support import support_node

    graph = StateGraph(AppState)

    # Every node is wrapped with observe_node() — this is the single place
    # where telemetry is applied so no individual node file needs to change.
    graph.add_node("router",              observe_node("router",              router_node))
    graph.add_node("research",            observe_node("research",            research_node))
    graph.add_node("support",             observe_node("support",             support_node))
    graph.add_node("retriever",           observe_node("retriever",           retriever_node))
    graph.add_node("reranker",            observe_node("reranker",            reranker_node))
    graph.add_node("generator",           observe_node("generator",           generator_node))
    graph.add_node("structured_output",   observe_node("structured_output",   structured_output_node))
    graph.add_node("groundedness",        observe_node("groundedness",        groundedness_node))
    graph.add_node("checkpoint",          observe_node("checkpoint",          checkpoint_node))
    graph.add_node("auto_approval_gate",  observe_node("auto_approval_gate",  auto_approval_gate_node))
    graph.add_node("human_approval",      observe_node("human_approval",      human_approval_node))
    graph.add_node("final_response",      observe_node("final_response",      final_response_node))
    graph.add_node("knowledge_update",    observe_node("knowledge_update",    knowledge_update_node))

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {"research": "research", "support": "support", END: END},
    )
    graph.add_edge("research", "retriever")
    graph.add_edge("support", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")
    graph.add_edge("generator", "structured_output")
    graph.add_edge("structured_output", "groundedness")
    graph.add_edge("groundedness", "checkpoint")
    graph.add_edge("checkpoint", "auto_approval_gate")
    graph.add_conditional_edges(
        "auto_approval_gate",
        _pre_approval_decision,
        {"final_response": "final_response", "human_approval": "human_approval"},
    )
    graph.add_conditional_edges(
        "human_approval",
        _approval_decision,
        {"final_response": "final_response", "human_approval": "human_approval", END: END},
    )
    graph.add_conditional_edges(
        "final_response",
        _post_final_decision,
        {"knowledge_update": "knowledge_update", END: END},
    )
    graph.add_edge("knowledge_update", END)

    return graph


def compile_workflow(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Compile the graph with an optional PostgreSQL checkpoint backend.

    interrupt_before=["human_approval"] causes LangGraph to persist state and
    pause execution at that node, waiting for an external resume() call.
    """
    graph = build_workflow()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"],
    )
