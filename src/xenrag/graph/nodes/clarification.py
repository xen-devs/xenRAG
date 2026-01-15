"""
AskClarification Node: Requests additional information from the user.
Explains why clarification is needed and what information is missing.
"""

from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from xenrag.graph.state import GraphState, ReasoningRecord
from xenrag.config import settings


async def clarification_node(state: GraphState) -> Dict[str, Any]:
    """
    Generates a clarification request when evidence is insufficient.
    Explains why clarification is needed and suggests what information would help.
    """
    print("--- ASK CLARIFICATION NODE ---")
    
    query = state.input_query
    context = state.retrieval_context
    emotion = state.emotion
    
    # Get reasoning about what's missing
    missing_context = "The retrieved information does not contain sufficient details to answer the question."
    if state.private_reasoning:
        for record in reversed(state.private_reasoning):
            summary = record.summary if hasattr(record, 'summary') else record.get('summary', '')
            if summary and 'missing' in summary.lower():
                missing_context = summary
                break
    
    tone_instruction = "Be polite and helpful."
    if emotion and emotion.type == "frustrated":
        tone_instruction = "Be extra apologetic and empathetic. Acknowledge their frustration."
    
    num_docs = len(context.merged_results) if context and context.merged_results else 0
    num_docs = len(context.merged_results) if context and context.merged_results else 0
    
    # Initialize LLM
    llm = ChatOllama(
        base_url=settings.OLLAMA_URL,
        model=settings.LLM_MODEL,
        temperature=0.5
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You need to ask the user for clarification because there isn't enough information to answer their question.

{tone_instruction}

Write a helpful message that:
1. Acknowledges their question
2. Explains you need more details
3. Suggests what specific information would help

Keep it conversational and helpful. Do NOT use JSON format - just write the message directly."""),
        ("user", """Original question: {query}

Why clarification is needed: {missing_context}

Number of related documents found: {num_docs}""")
    ])
    
    chain = prompt | llm
    
    try:
        response_msg = await chain.ainvoke({
            "query": query,
            "missing_context": missing_context,
            "num_docs": num_docs,
            "tone_instruction": tone_instruction
        })
        
        clarification_message = response_msg.content.strip()
        
        if clarification_message.startswith('"') and clarification_message.endswith('"'):
            clarification_message = clarification_message[1:-1]
        
        print(f"Clarification requested: {len(clarification_message)} chars")
        
        return {
            "clarification_message": clarification_message,
            "clarification_reason": missing_context,
            "needs_clarification": True,
            "private_reasoning": [
                ReasoningRecord(
                    step="AskClarification",
                    summary=f"Requested clarification due to insufficient evidence. Found {num_docs} docs but none directly address the question.",
                    confidence=1.0
                )
            ]
        }
        
    except Exception as e:
        print(f"Clarification Error: {e}")
        
        # Fallback message
        fallback_msg = f"I found {num_docs} related reviews, but I need a bit more context to give you a helpful answer. Could you please:\n\n"
        fallback_msg += "• Be more specific about what aspect you're interested in?\n"
        fallback_msg += "• Mention any particular product or feature?\n"
        fallback_msg += "• Tell me what kind of information would be most helpful?"
        
        return {
            "clarification_message": fallback_msg,
            "clarification_reason": "Insufficient context to generate accurate response",
            "needs_clarification": True,
            "private_reasoning": [
                ReasoningRecord(
                    step="AskClarification",
                    summary=f"Used fallback clarification. Error: {e}",
                    confidence=0.5
                )
            ]
        }
