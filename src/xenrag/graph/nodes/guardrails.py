"""
Guardrail Nodes for LangGraph integration.
"""

from typing import Dict, Any
from xenrag.graph.state import GraphState, ReasoningRecord
from xenrag.guardrails.input_rail import validate_input
from xenrag.guardrails.topic_rail import validate_topic
from xenrag.guardrails.output_rail import validate_output
from xenrag.guardrails.retrieval_rail import filter_retrieval_results


async def input_guardrail_node(state: GraphState) -> Dict[str, Any]:
    """
    Input guardrail node: Validates input before processing.
    Checks for jailbreaks, toxic content, and off-topic queries.
    """
    print("--- INPUT GUARDRAIL NODE ---")
    
    query = state.input_query
    conversation_history = state.conversation_history
    pending_clarification = state.pending_clarification
    warnings = []
    
    # Validate input safety
    input_result = validate_input(query)
    
    if not input_result.is_safe:
        print(f"Input blocked: {input_result.risk_type}")
        return {
            "is_blocked": True,
            "blocked_reason": input_result.blocked_reason,
            "pending_clarification": False,
            "private_reasoning": [
                ReasoningRecord(
                    step="InputGuardrail",
                    summary=f"Blocked: {input_result.risk_type}",
                    confidence=1.0
                )
            ]
        }
    
    # Check topic relevance with conversation context
    topic_result = validate_topic(
        query, 
        pending_clarification=pending_clarification,
        conversation_history=conversation_history
    )
    
    if not topic_result.is_on_topic:
        print(f"Off-topic: {topic_result.off_topic_category}")
        return {
            "is_blocked": True,
            "blocked_reason": topic_result.redirect_message,
            "pending_clarification": False,
            "private_reasoning": [
                ReasoningRecord(
                    step="InputGuardrail",
                    summary=f"Off-topic: {topic_result.off_topic_category}",
                    confidence=1.0
                )
            ]
        }
    
    # Collect warnings
    if input_result.pii_detected:
        warnings.append(f"PII detected: {', '.join(input_result.pii_detected)}")
    
    print(f"Input validated: safe=True, topic_confidence={topic_result.topic_confidence:.2f}")
    
    return {
        "input_query": input_result.sanitized_query or query,
        "guardrail_warnings": warnings,
        "private_reasoning": [
            ReasoningRecord(
                step="InputGuardrail",
                summary=f"Passed validation. Topic confidence: {topic_result.topic_confidence:.2f}",
                confidence=topic_result.topic_confidence
            )
        ]
    }


async def output_guardrail_node(state: GraphState) -> Dict[str, Any]:
    """
    Output guardrail node: Validates response before returning to user.
    """
    print("--- OUTPUT GUARDRAIL NODE ---")
    
    response = state.generated_answer
    
    if not response:
        return {}
    
    # Get source docs for hallucination check
    source_docs = []
    if state.retrieval_context and state.retrieval_context.merged_results:
        source_docs = [r.content for r in state.retrieval_context.merged_results]
    
    # Validate output
    result = validate_output(response, source_docs)
    
    if not result.is_safe:
        print(f"Output blocked: {result.blocked_reason}")
        return {
            "generated_answer": result.modified_response,
            "private_reasoning": [
                ReasoningRecord(
                    step="OutputGuardrail",
                    summary=f"Response modified: {result.blocked_reason}",
                    confidence=1.0
                )
            ]
        }
    
    # Apply modifications and warnings
    warnings = state.guardrail_warnings.copy() if state.guardrail_warnings else []
    warnings.extend(result.warnings)
    
    print(f"Output validated: confidence={result.confidence_score:.2f}")
    
    return {
        "generated_answer": result.modified_response,
        "guardrail_warnings": warnings,
        "private_reasoning": [
            ReasoningRecord(
                step="OutputGuardrail",
                summary=f"Validated. Confidence: {result.confidence_score:.2f}. Warnings: {len(result.warnings)}",
                confidence=result.confidence_score
            )
        ]
    }
