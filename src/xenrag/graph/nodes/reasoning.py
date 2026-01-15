"""
Reason Node: Evaluates evidence sufficiency.
This node does NOT generate text - it only decides if we have enough evidence to proceed.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from xenrag.graph.state import GraphState, ReasoningRecord
from xenrag.llm.langchain_wrapper import get_managed_llm


class SufficiencyOutput(BaseModel):
    is_sufficient: bool = Field(..., description="True if the provided documents contain enough information to answer the query.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the sufficiency assessment (0-1).")
    missing_information: str = Field(..., description="What specific information is missing, if not sufficient.")
    reasoning: str = Field(..., description="Brief explanation of the sufficiency decision.")


async def reasoning_node(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates if the retrieved context is sufficient to answer the query.
    Does NOT generate the answer - only evaluates evidence sufficiency.
    
    Routes to:
    - GenerateAnswer Node if sufficient
    - AskClarification Node if insufficient
    """
    print("--- REASON NODE ---")
    query = state.input_query
    context = state.retrieval_context
    
    # Check if we have any context at all
    if not context or not context.merged_results:
        print("No documents retrieved - insufficient evidence.")
        return {
            "is_sufficient": False,
            "needs_clarification": True,
            "private_reasoning": [
                ReasoningRecord(
                    step="Reasoner",
                    summary="No documents retrieved. Cannot evaluate sufficiency.",
                    confidence=1.0
                )
            ]
        }

    docs_text = "\n\n".join([
        f"Doc {i+1} [{item.source}]:\n{item.content}" 
        for i, item in enumerate(context.merged_results)
    ])
    
    # Use managed LLM with failover
    llm = get_managed_llm(temperature=0)
    
    parser = JsonOutputParser(pydantic_object=SufficiencyOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict evidence evaluator. Your ONLY job is to determine if the provided documents contain sufficient information to answer the user's question.

Rules:
1. Do NOT answer the question yourself.
2. Evaluate ONLY whether the documents contain the necessary information.
3. Be conservative - if you're unsure, mark as insufficient.
4. Consider: Does the context directly address the question? Is there enough detail?

Sufficiency threshold:
- SUFFICIENT: The context directly addresses the question with enough detail to form a complete answer.
- INSUFFICIENT: The context is missing key information, is tangential, or doesn't address the question.

Return JSON only.
{format_instructions}
"""),
        ("user", "User Query: {query}\n\nAvailable Context:\n{context}")
    ])
    
    chain = prompt | llm
    
    try:
        response_msg = await chain.ainvoke({
            "query": query, 
            "context": docs_text,
            "format_instructions": parser.get_format_instructions()
        })
        
        result = parser.parse(response_msg.content)
        output = SufficiencyOutput(**result)
        
        print(f"Sufficiency: {output.is_sufficient} (confidence: {output.confidence:.2f})")
        if not output.is_sufficient:
            print(f"Missing: {output.missing_information}")
        
        return {
            "is_sufficient": output.is_sufficient,
            "needs_clarification": not output.is_sufficient,
            "private_reasoning": [
                ReasoningRecord(
                    step="Reasoner",
                    summary=f"{output.reasoning}. Missing: {output.missing_information}" if not output.is_sufficient else output.reasoning,
                    confidence=output.confidence
                )
            ]
        }
        
    except Exception as e:
        print(f"Reasoner Error: {e}")
        return {
            "is_sufficient": False,
            "needs_clarification": True,
            "private_reasoning": [
                ReasoningRecord(
                    step="Reasoner",
                    summary=f"Error during evaluation: {e}",
                    confidence=0.0
                )
            ]
        }
