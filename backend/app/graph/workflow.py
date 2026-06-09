from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.state import WorkflowState


def _route_decision(state: WorkflowState) -> str:
    """Conditional edge: direct flow after routing node."""
    if state.error:
        return END
    return state.route or END


def _approval_decision(state: WorkflowState) -> str:
    """Conditional edge: proceed or halt after human approval."""
    if state.approval_status == "approved":
        return "final_response"
    if state.approval_status == "rejected":
        return END
    return "human_approval"


def build_workflow() -> StateGraph:
    """Construct the LangGraph state machine (not yet compiled)."""
    from app.graph.nodes.router import router_node
    from app.graph.nodes.research import research_node
    from app.graph.nodes.support import support_node
    from app.graph.nodes.retriever import retriever_node
    from app.graph.nodes.reranker import reranker_node
    from app.graph.nodes.generator import generator_node
    from app.graph.nodes.structured_output import structured_output_node
    from app.graph.nodes.checkpoint import checkpoint_node
    from app.graph.nodes.human_approval import human_approval_node
    from app.graph.nodes.final_response import final_response_node

    graph = StateGraph(WorkflowState)

    graph.add_node("router", router_node)
    graph.add_node("research", research_node)
    graph.add_node("support", support_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("generator", generator_node)
    graph.add_node("structured_output", structured_output_node)
    graph.add_node("checkpoint", checkpoint_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("final_response", final_response_node)

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
    graph.add_edge("structured_output", "checkpoint")
    graph.add_edge("checkpoint", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        _approval_decision,
        {"final_response": "final_response", "human_approval": "human_approval", END: END},
    )
    graph.add_edge("final_response", END)

    return graph


def compile_workflow(checkpointer: Any | None = None) -> CompiledStateGraph:
    graph = build_workflow()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"],
    )
