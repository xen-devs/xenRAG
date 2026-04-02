"""
Chat Service — CRUD operations for chats and messages.
"""

import uuid
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.db.models.chat import Chat, Message
from api.db.models.knowledge_base_dataset import KnowledgeBaseDataset


class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_chat(
        self, org_id: str, user_id: str, title: Optional[str] = None
    ) -> Chat:
        chat = Chat(
            id=uuid.uuid4(),
            org_id=uuid.UUID(org_id),
            user_id=uuid.UUID(user_id),
            title=title,
        )
        self.db.add(chat)
        await self.db.flush()
        return chat

    async def add_message(self, chat_id: str, role: str, content: str) -> Message:
        msg = Message(
            id=uuid.uuid4(),
            chat_id=uuid.UUID(chat_id),
            role=role,
            content=content,
        )
        self.db.add(msg)
        await self.db.flush()
        return msg

    async def get_chat(self, chat_id: str, org_id: str) -> Optional[Chat]:
        result = await self.db.execute(
            select(Chat)
            .options(selectinload(Chat.messages))
            .where(Chat.id == uuid.UUID(chat_id), Chat.org_id == uuid.UUID(org_id))
        )
        return result.scalar_one_or_none()

    async def list_chats(self, org_id: str, user_id: str) -> List[Chat]:
        result = await self.db.execute(
            select(Chat)
            .where(Chat.org_id == uuid.UUID(org_id), Chat.user_id == uuid.UUID(user_id))
            .order_by(Chat.updated_at.desc())
        )
        return list(result.scalars().all())

    async def update_chat_title(self, chat_id: str, title: str) -> None:
        await self.db.execute(
            update(Chat).where(Chat.id == uuid.UUID(chat_id)).values(title=title)
        )
        await self.db.flush()

    async def delete_chat(self, chat_id: str) -> bool:
        result = await self.db.execute(
            select(Chat).where(Chat.id == uuid.UUID(chat_id))
        )
        chat = result.scalar_one_or_none()
        if not chat:
            return False
        await self.db.delete(chat)
        await self.db.flush()
        return True

    async def get_active_collection(self, org_id: str) -> Optional[str]:
        """Get the active KB dataset collection name for an org."""
        result = await self.db.execute(
            select(KnowledgeBaseDataset.collection_name).where(
                KnowledgeBaseDataset.org_id == uuid.UUID(org_id),
                KnowledgeBaseDataset.is_active == True,
            )
        )
        return result.scalar_one_or_none()