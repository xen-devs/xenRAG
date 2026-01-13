"""
Graph construction and compilation.
"""

from langgraph.graph import StateGraph, END, START
from xenrag.graph.state import GraphState
from xenrag.graph.nodes.interpreter import interpreter_node


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

    # Add edges
    workflow.add_edge(START, "interpreter")
    workflow.add_edge("interpreter", END)

    # Compile the graph
    return workflow.compile()
