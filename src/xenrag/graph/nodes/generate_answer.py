"""
GenerateAnswer Node: Generates the final answer using retrieved context.
Adjusts tone based on detected user emotion.
"""

from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from xenrag.graph.state import GraphState, ReasoningRecord
from xenrag.config import settings


async def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """
    Generates the final answer from retrieved context.
    Adjusts tone based on the detected user emotion.
    """
    print("--- GENERATE ANSWER NODE ---")
    
    query = state.input_query
    context = state.retrieval_context
    emotion = state.emotion
    
    if context and context.merged_results:
        docs_text = "\n\n".join([
            f"[Source: {item.source}] {item.content}" 
            for item in context.merged_results
        ])
    else:
        docs_text = "No context available."
    
    tone_name = "neutral"
    tone_instruction = "Use a neutral, professional tone."
    if emotion:
        if emotion.type == "frustrated":
            tone_name = "empathetic"
            tone_instruction = "Use an empathetic and understanding tone. Acknowledge the user's frustration."
        elif emotion.type == "happy":
            tone_name = "enthusiastic"
            tone_instruction = "Use an enthusiastic and positive tone."
        elif emotion.type == "confused":
            tone_name = "clear"
            tone_instruction = "Use a clear, simple, and helpful tone. Avoid jargon."
    
    # Initialize LLM
    llm = ChatOllama(
        base_url=settings.OLLAMA_URL,
        model=settings.LLM_MODEL,
        temperature=0.7
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful customer support assistant analyzing customer reviews.

Your task is to answer the user's question based ONLY on the provided context.
Do not use information outside the context. If the context doesn't contain relevant information, say so.

{tone_instruction}

Provide a clear, helpful answer. Do NOT wrap your response in JSON or any special format - just provide the answer directly."""),
        ("user", "Question: {query}\n\nContext from customer reviews:\n{context}")
    ])
    
    chain = prompt | llm
    
    try:
        response_msg = await chain.ainvoke({
            "query": query,
            "context": docs_text,
            "tone_instruction": tone_instruction
        })
        
        answer = response_msg.content.strip()
        
        if answer.startswith('```') and answer.endswith('```'):
            answer = answer[3:-3].strip()
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        
        print(f"Generated answer with {tone_name} tone ({len(answer)} chars)")
        
        return {
            "generated_answer": answer,
            "private_reasoning": [
                ReasoningRecord(
                    step="GenerateAnswer",
                    summary=f"Generated response using {tone_name} tone based on {emotion.type if emotion else 'neutral'} emotion.",
                    confidence=1.0
                )
            ]
        }
        
    except Exception as e:
        print(f"GenerateAnswer Error: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            simple_prompt = f"Based on this context about customer reviews:\n{docs_text[:2000]}\n\nAnswer this question: {query}"
            response = await llm.ainvoke(simple_prompt)
            return {
                "generated_answer": response.content.strip(),
                "private_reasoning": [
                    ReasoningRecord(
                        step="GenerateAnswer",
                        summary="Used fallback simple generation.",
                        confidence=0.7
                    )
                ]
            }
        except Exception as e2:
            return {
                "generated_answer": f"Based on the available customer reviews, I found some relevant information but encountered an issue formatting the response. The context contains {len(context.merged_results) if context and context.merged_results else 0} relevant documents.",
                "private_reasoning": [
                    ReasoningRecord(
                        step="GenerateAnswer",
                        summary=f"Error during generation: {e}. Fallback also failed: {e2}",
                        confidence=0.0
                    )
                ]
            }
