"""
Graph construction and compilation.
"""

import atexit
from typing import Optional

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from ai_core.graph.state import GraphState
from ai_core.graph.nodes.interpreter import interpreter_node
from ai_core.graph.nodes.query import query_node
from ai_core.graph.nodes.reasoning import reasoning_node
from ai_core.graph.nodes.generate_answer import generate_answer_node
from ai_core.graph.nodes.clarification import clarification_node
from ai_core.graph.nodes.explanation import explanation_node
from ai_core.graph.nodes.guardrails import input_guardrail_node, output_guardrail_node
from ai_core.config import settings

_pool: Optional[AsyncConnectionPool] = None


def _cleanup_pool():
    """Schedule pool close on process exit."""
    global _pool
    if _pool is not None:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_pool.close())
            else:
                loop.run_until_complete(_pool.close())
        except Exception:
            pass
        _pool = None


atexit.register(_cleanup_pool)


def should_continue_after_input_guard(state: GraphState) -> str:
    """Route based on input guardrail result."""
    if state.is_blocked:
        return "blocked"
    return "continue"


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


_checkpointer: Optional[AsyncPostgresSaver] = None
def build_graph():
    """
    Builds and compiles the StateGraph.

    Returns:
        CompiledStateGraph: The compiled graph ready for execution.
    """
    global _pool, _checkpointer

    checkpointer = None
    if settings.DATABASE_URL_SYNC:
        try:
            _pool = AsyncConnectionPool(
                conninfo=settings.DATABASE_URL_SYNC,
                max_size=5,
                open=False,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            checkpointer = AsyncPostgresSaver(conn=_pool)
            _checkpointer = checkpointer
            print("[Checkpointer] Async PostgreSQL checkpointer initialized.")
        except Exception as e:
            print(f"[Checkpointer] Failed to initialize checkpointer: {e}")
            checkpointer = None
    else:
        print("[Checkpointer] DATABASE_URL_SYNC not set - no conversation memory.")

    # Initialize the graph with the state schema
    workflow = StateGraph(GraphState)

    # Add nodes
    # workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("interpreter", interpreter_node)
    workflow.add_node("query", query_node)
    workflow.add_node("reasoner", reasoning_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("output_guardrail", output_guardrail_node)
    workflow.add_node("ask_clarification", clarification_node)
    workflow.add_node("build_explanation", explanation_node)

    # Add edges
    workflow.add_edge(START, "interpreter")

    # workflow.add_conditional_edges(
    #     "input_guardrail",
    #     should_continue_after_input_guard,
    #     {
    #         "blocked": END,
    #         "continue": "interpreter"
    #     }
    # )
    
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
    
    workflow.add_edge("generate_answer", "output_guardrail")
    workflow.add_edge("output_guardrail", "build_explanation")
    workflow.add_edge("build_explanation", END)
    
    workflow.add_edge("ask_clarification", END)

    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)
async def setup_checkpointer():
    if _pool is not None:
        await _pool.open()
    if _checkpointer is not None:
        await _checkpointer.setup()
        print("[Checkpointer] Tables created successfully.")
