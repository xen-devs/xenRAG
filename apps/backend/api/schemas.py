from typing import List, Optional

from pydantic import BaseModel, Field


class ConversationMessageInput(BaseModel):
    role: str = Field(..., description="Role: user, assistant, or clarification")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")


class ChatRequest(BaseModel):
    input_query: str = Field(..., min_length=1, description="User query text")
    conversation_history: List[ConversationMessageInput] = Field(
        default_factory=list,
        description="Recent conversation turns",
    )
    pending_clarification: bool = Field(
        False, description="Whether previous turn asked for clarification"
    )


class ChatResponse(BaseModel):
    intent: Optional[dict] = None
    emotion: Optional[dict] = None
    is_blocked: bool = False
    blocked_reason: str = ""
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    clarification_reason: Optional[str] = None
    generated_answer: Optional[str] = None
    explanations: List[dict] = Field(default_factory=list)
    private_reasoning: List[dict] = Field(default_factory=list)
