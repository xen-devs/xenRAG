"""
Interpreter Node: Classifies user intent and emotion.
"""

from typing import Dict, Any, cast
from xenrag.graph.state import GraphState, Intent, Emotion, ReasoningRecord
from xenrag.config.settings import OLLAMA_URL, LLM_MODEL
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import json
import re

def _extract_json(content: str) -> Dict[str, Any]:
    """
    Robust JSON extraction from LLM output.
    Handling Markdown code blocks and raw JSON.
    """
    # Find JSON within code blocks first
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not json_match:
        # Find raw JSON block
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1) if json_match.lastindex == 1 else json_match.group(0)
        return json.loads(json_str)
    
    raise ValueError(f"No JSON found in output: {content}")

async def interpreter_node(state: GraphState) -> Dict[str, Any]:
    """
    Analyzes the user's input to determine intent and emotion.
    Async implementation for high-concurrency production usage.
    """
    print(f"--- INTERPRETER NODE (Model: {LLM_MODEL}) ---")
    query = state.input_query
    
    # Initialize Ollama Chat Model
    llm = ChatOllama(
        base_url=OLLAMA_URL,
        model=LLM_MODEL,
        temperature=0,
    )
    
    system_prompt = """
    You are an expert intent and emotion classifier for a customer review analysis system.
    
    Analyze the user's query and classify:
    1. Intent: What does the user want? (complaint_analysis, why_explanation, summary_request, follow_up, clarification_request, etc.)
    2. Emotion: What is the user's tone? (neutral, frustrated, confused, satisfied, etc.)
    3. Confidence: How sure are you? (0.0 to 1.0)
    
    Provide brief reasoning for your decision.
    
    Return a JSON object with the following structure:
    {{
        "intent": {{
            "type": "intent_type",
            "confidence": 0.95
        }},
        "emotion": {{
            "type": "emotion_type",
            "confidence": 0.95
        }},
        "reasoning": "reasoning text"
    }}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    try:
        # Invoke LLM
        result_msg = await llm.ainvoke(messages)
        content = result_msg.content
        
        # Extract JSON
        result = _extract_json(content)
        

        intent_data = result.get("intent", {})
        emotion_data = result.get("emotion", {})
        
        intent = Intent(
            type=intent_data.get("type", "unknown"),
            confidence=float(intent_data.get("confidence", 0.0))
        )
        
        emotion = Emotion(
            type=emotion_data.get("type", "neutral"),
            confidence=float(emotion_data.get("confidence", 0.0))
        )
        
        reasoning = result.get("reasoning", "No reasoning provided.")

        record = ReasoningRecord(
            step="interpreter",
            summary=f"Classified intent as '{intent.type}' and emotion as '{emotion.type}'. Reasoning: {reasoning}",
            confidence=min(intent.confidence, emotion.confidence)
        )
        
        state_update = {
            "intent": intent,
            "emotion": emotion,
            "private_reasoning": [record]
        }
        print(f"--- INTERPRETER NODE OUTPUT ---\n{state_update}")
        return state_update

    except Exception as e:
        print(f"Error in interpreter node: {e}")
        
        fallback_intent = Intent(type="unknown", confidence=0.0)
        fallback_emotion = Emotion(type="neutral", confidence=0.0)
        
        record = ReasoningRecord(
            step="interpreter",
            summary=f"Failed to classify. Error: {str(e)}",
            confidence=0.0
        )
         
        state_update = {
            "intent": fallback_intent,
            "emotion": fallback_emotion,
            "private_reasoning": [record]
        }
        print(f"--- INTERPRETER NODE ERROR OUTPUT ---\n{state_update}")
        return state_update
