"""
Graph construction and compilation.
"""

from langgraph.graph import StateGraph, END, START
from xenrag.graph.state import GraphState
from xenrag.graph.nodes.interpreter import interpreter_node
from xenrag.graph.nodes.query import query_node
from xenrag.graph.nodes.reasoning import reasoning_node
from xenrag.graph.nodes.generate_answer import generate_answer_node
from xenrag.graph.nodes.clarification import clarification_node
from xenrag.graph.nodes.explanation import explanation_node


def should_generate_or_clarify(state: GraphState) -> str:
    """
    Routing function: Determines whether to generate answer or ask for clarification.
    
    Returns:
        "generate_answer" if evidence is sufficient
        "ask_clarification" if clarification is needed
    """
    if state.is_sufficient:
        return "generate_answer"
    else:
        return "ask_clarification"


def build_graph() -> StateGraph:
    """
    Builds and compiles the StateGraph.

    Returns:
        CompiledStateGraph: The compiled graph ready for execution.
    """
    # Initialize the graph with the state schema
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("interpreter", interpreter_node)
    workflow.add_node("query", query_node)
    workflow.add_node("reasoner", reasoning_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("ask_clarification", clarification_node)
    workflow.add_node("build_explanation", explanation_node)

    # Add edges
    workflow.add_edge(START, "interpreter")
    workflow.add_edge("interpreter", "query")
    workflow.add_edge("query", "reasoner")
    
    workflow.add_conditional_edges(
        "reasoner",
        should_generate_or_clarify,
        {
            "generate_answer": "generate_answer",
            "ask_clarification": "ask_clarification"
        }
    )
    
    workflow.add_edge("generate_answer", "build_explanation")
    workflow.add_edge("build_explanation", END)
    
    workflow.add_edge("ask_clarification", END)

    # Compile the graph
    return workflow.compile()
