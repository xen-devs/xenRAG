"""
State definition for the LangGraph application.
Supports advanced validation, hybrid retrieval, and explainability without CoT leakage.
"""

from typing import Annotated, List, Optional, Dict
import operator
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class Intent(BaseModel):
    """Classification of the user's intent."""
    type: str = Field(..., description="The classified intent type (e.g., specific_question, explanation).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the classification.")


class Emotion(BaseModel):
    """Sentiment and tone analysis of the user input."""
    type: str = Field(..., description="The detected emotion (e.g., neutral, frustrated).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the detection.")


class RetrievalItem(BaseModel):
    """A single unit of retrieved context."""
    id: str = Field(..., description="Unique identifier for the item.")
    content: str = Field(..., description="The content of the retrieved item.")
    source: str = Field(..., description="Source of the item (e.g., document name, graph node).")
    score: float = Field(..., description="Relevance score.")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata.")


class RetrievalContext(BaseModel):
    """Container for hybrid retrieval results."""
    vector_results: List[RetrievalItem] = Field(default_factory=list, description="Results from vector search.")
    kg_results: List[RetrievalItem] = Field(default_factory=list, description="Results from knowledge graph.")
    merged_results: List[RetrievalItem] = Field(default_factory=list, description="Unified list of results for generation.")
    retrieval_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in the retrieval.")


class Explanation(BaseModel):
    """Structured metadata for explanation, separate from generation."""
    reasoning_type: str = Field(..., description="Type of reasoning used (e.g., deduction, analogy).")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of RetrievalItems used as evidence.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this explanation step.")
    limitations: Optional[str] = Field(None, description="Known limitations or caveats.")


class ReasoningRecord(BaseModel):
    """Internal reasoning step record. NOT for user display."""
    step: str = Field(..., description="Name of the reasoning step.")
    summary: str = Field(..., description="Structured summary of the outcome.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this step.")


class GraphState(BaseModel):
    """
    Graph State.
    """
    input_query: str = Field(..., description="Original user query.")

    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    
    intent: Optional[Intent] = Field(None, description="Detected intent.")
    emotion: Optional[Emotion] = Field(None, description="Detected emotion.")
    
    retrieval_context: Optional[RetrievalContext] = Field(None, description="Hybrid retrieval context.")
    
    is_sufficient: bool = Field(False, description="Whether retrieved context is sufficient.")
    needs_clarification: bool = Field(False, description="Whether user clarification is needed.")
    
    generated_answer: Optional[str] = Field(None, description="Final generated answer.")
    
    explanations: List[Explanation] = Field(default_factory=list, description="Explainable metadata.")
    
    private_reasoning: List[ReasoningRecord] = Field(default_factory=list, description="Internal reasoning trace.")
