"""
BuildExplanation Node: Constructs structured explainability artifacts.
Maps claims to evidence, provides confidence scores, and notes limitations.
"""

from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from xenrag.graph.state import GraphState, Explanation, ReasoningRecord
from xenrag.config import settings
from xenrag.utils.json_parser import parse_json_safe


async def explanation_node(state: GraphState) -> Dict[str, Any]:
    """
    Builds structured explainability artifacts.
    Maps claims to evidence and provides confidence assessment.
    """
    print("--- BUILD EXPLANATION NODE ---")
    
    query = state.input_query
    answer = state.generated_answer
    context = state.retrieval_context
    
    if not answer:
        print("No answer to explain.")
        return {
            "explanations": [],
            "private_reasoning": [
                ReasoningRecord(
                    step="BuildExplanation",
                    summary="No answer available to build explanation.",
                    confidence=0.0
                )
            ]
        }
    
    source_ids = []
    sources_summary = []
    if context and context.merged_results:
        for i, item in enumerate(context.merged_results):
            source_id = item.id or f"doc_{i}"
            source_ids.append(source_id)
            content_preview = item.content[:100] + "..." if len(item.content) > 100 else item.content
            sources_summary.append(f"{source_id}: {content_preview}")
    
    # Initialize LLM
    llm = ChatOllama(
        base_url=settings.OLLAMA_URL,
        model=settings.LLM_MODEL,
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an explainability analyst. Analyze the AI-generated answer and explain how it relates to the source documents.

Provide your analysis in this exact JSON format:
{{
    "reasoning_type": "synthesis|deduction|comparison|summarization",
    "confidence": 0.0 to 1.0,
    "summary": "Brief summary of what the answer covers",
    "limitations": "Any limitations or caveats"
}}

Rules:
- confidence: 1.0 if answer directly uses sources, 0.7 if inferred, 0.5 if partially supported
- reasoning_type: how the answer was constructed from sources
- limitations: what the answer doesn't cover or uncertainties

Return ONLY the JSON object, no other text."""),
        ("user", """Question: {query}

Answer: {answer}

Source IDs used: {source_ids}""")
    ])
    
    chain = prompt | llm
    
    try:
        response_msg = await chain.ainvoke({
            "query": query,
            "answer": answer[:500],
            "source_ids": ", ".join(source_ids[:5]) if source_ids else "none"
        })
        
        result = parse_json_safe(response_msg.content)
        
        if result:
            reasoning_type = result.get("reasoning_type", "synthesis")
            confidence = float(result.get("confidence", 0.7))
            summary = result.get("summary", "Answer synthesized from customer reviews.")
            limitations = result.get("limitations", "Based on available review data only.")
        else:
            # Fallback values
            reasoning_type = "synthesis"
            confidence = 0.7
            summary = "Answer synthesized from customer reviews."
            limitations = "Unable to generate detailed explanation."
        
        print(f"Built explanation: {reasoning_type}, confidence: {confidence:.2f}")
        
        explanation = Explanation(
            reasoning_type=reasoning_type,
            evidence_ids=source_ids[:5],
            confidence=confidence,
            limitations=limitations
        )
        
        return {
            "explanations": [explanation],
            "private_reasoning": [
                ReasoningRecord(
                    step="BuildExplanation",
                    summary=f"Reasoning: {reasoning_type}. {summary}. Limitations: {limitations}",
                    confidence=confidence
                )
            ]
        }
        
    except Exception as e:
        print(f"Explanation Error: {e}")
        
        return {
            "explanations": [
                Explanation(
                    reasoning_type="synthesis",
                    evidence_ids=source_ids[:5] if source_ids else [],
                    confidence=0.6,
                    limitations="Explanation generated from available context."
                )
            ],
            "private_reasoning": [
                ReasoningRecord(
                    step="BuildExplanation",
                    summary=f"Generated fallback explanation. Error: {e}",
                    confidence=0.6
                )
            ]
        }
