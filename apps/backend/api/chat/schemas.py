from typing import List, Optional
from pydantic import BaseModel, Field


class SendMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message text")
    chat_id: Optional[str] = Field(
        None, description="Existing chat ID, or null to create new chat"
    )


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


class ChatOut(BaseModel):
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str


class ChatDetailOut(ChatOut):
    messages: List[MessageOut]


class RenameChatRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ChatListResponse(BaseModel):
    chats: List[ChatOut]