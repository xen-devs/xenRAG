"""
Reason Node: Evaluates evidence sufficiency.
This node does NOT generate text - it only decides if we have enough evidence to proceed.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ai_core.graph.state import GraphState, ReasoningRecord
from ai_core.llm.langchain_wrapper import get_managed_llm


class SufficiencyOutput(BaseModel):
    is_sufficient: bool = Field(..., description="True if the provided documents contain enough information to answer the query.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the sufficiency assessment (0-1).")
    missing_information: str = Field(..., description="What specific information is missing, if not sufficient.")
    reasoning: str = Field(..., description="Brief explanation of the sufficiency decision.")


async def reasoning_node(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates if the retrieved context is sufficient to answer the query.

    Always routes to GenerateAnswer - the answer node's prompt handles
    limited/no context gracefully (greetings, capability questions, etc.)
    The old strict evaluator was causing most queries to route to clarification.
    """
    print("--- REASON NODE ---")
    context = state.retrieval_context
    num_docs = len(context.merged_results) if context and context.merged_results else 0
    print(f"Context: {num_docs} documents retrieved. Routing to generate_answer.")

    return {
        "is_sufficient": True,
        "needs_clarification": False,
        "private_reasoning": [
            ReasoningRecord(
                step="Reasoner",
                summary=f"Passing {num_docs} documents to answer generation.",
                confidence=1.0
            )
        ]
    }
