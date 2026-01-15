"""
Interpreter Node: Classifies user intent and emotion.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from xenrag.graph.state import GraphState, Intent, Emotion
from xenrag.llm.langchain_wrapper import get_managed_llm


class InterpretationOutput(BaseModel):
    intent_type: str = Field(..., description="Classification of user intent: ['complaint_analysis', 'specific_question', 'summary_request', 'feature_request', 'unknown']")
    intent_confidence: float = Field(..., description="0.0 to 1.0 confidence score for intent")
    emotion_type: str = Field(..., description="User emotion: ['neutral', 'frustrated', 'happy', 'confused']")
    emotion_confidence: float = Field(..., description="0.0 to 1.0 confidence score for emotion")

async def interpreter_node(state: GraphState) -> Dict[str, Any]:
    """
    Analyzes the user's input to determine Intent and Emotion.
    """
    print("--- INTERPRETER NODE ---")
    query = state.input_query
    
    # Use managed LLM with failover
    llm = get_managed_llm(temperature=0)
    
    parser = JsonOutputParser(pydantic_object=InterpretationOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert intent and emotion classifier for a customer review analysis system.
        Analyze the user's query and classify it into one of the following intents:
        - complaint_analysis: User wants to know about problems or bad reviews.
        - specific_question: User asks about a specific attribute (e.g., "battery life", "price").
        - summary_request: User wants a general overview.
        - feature_request: User asks about a missing feature.
        - unknown: Cannot determine.

        Also classify the user's emotion as: neutral, frustrated, happy, or confused.

        Return ONLY a JSON object matching the requested format.
        {format_instructions}
        """),
        ("user", "{query}")
    ])
    
    chain = prompt | llm
    
    try:
        response_msg = await chain.ainvoke({"query": query, "format_instructions": parser.get_format_instructions()})
        
        result = parser.parse(response_msg.content)
        
        intent = Intent(
            type=result["intent_type"],
            confidence=result["intent_confidence"]
        )
        
        emotion = Emotion(
            type=result["emotion_type"],
            confidence=result["emotion_confidence"]
        )
        
        print(f"Detected: {intent.type} ({intent.confidence:.2f}), {emotion.type} ({emotion.confidence:.2f})")
        
        return {
            "intent": intent, 
            "emotion": emotion,
            "private_reasoning": [
                {
                    "step": "Interpreter",
                    "summary": f"Classified as {intent.type} with {emotion.type} emotion.",
                    "confidence": min(intent.confidence, emotion.confidence)
                }
            ]
        }
        
    except Exception as e:
        print(f"Interpreter Error: {e}")
        return {
            "intent": Intent(type="specific_question", confidence=0.5),
            "emotion": Emotion(type="neutral", confidence=0.5)
        }
