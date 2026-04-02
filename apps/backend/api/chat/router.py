"""
Chat API Router
All endpoints are org-scoped: /api/v1/orgs/{org_id}/chats/...
"""

import json
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import get_current_user
from api.db.models import User
from api.db.models.chat import Chat, Message
from api.db.session import async_session, get_session
from ai_core.llm.langchain_wrapper import get_managed_llm
from api.chat.schemas import (
    ChatDetailOut,
    ChatListResponse,
    ChatOut,
    MessageOut,
    RenameChatRequest,
    SendMessageRequest,
)
from api.chat.service import ChatService
from ai_core.graph.graph import build_graph
from ai_core.graph.state import ConversationMessage

logger = logging.getLogger(__name__)

chat_router = APIRouter(
    prefix="/orgs/{org_id}/chats",
    tags=["chat"],
)

_graph = build_graph()

NODE_STATUS_MAP = {
    "input_guardrail": "Checking your query...",
    "interpreter": "Understanding your question...",
    "query": "Searching knowledge base...",
    "reasoner": "Analyzing results...",
    "generate_answer": "Generating answer...",
    "output_guardrail": "Validating response...",
    "ask_clarification": "Preparing clarification...",
    "build_explanation": "Building explanation...",
}


def _check_org_membership(user: User, org_id: str) -> None:
    for m in user.memberships:
        if str(m.org_id) == org_id:
            return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You do not have access to this organization",
    )


def _model_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_model_to_dict(item) for item in value]
    return value


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@chat_router.get("", response_model=ChatListResponse)
async def list_chats(
    org_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = ChatService(db)
    chats = await svc.list_chats(org_id, str(current_user.id))
    return ChatListResponse(
        chats=[
            ChatOut(
                id=str(c.id),
                title=c.title,
                created_at=c.created_at.isoformat(),
                updated_at=c.updated_at.isoformat(),
            )
            for c in chats
        ]
    )


@chat_router.get("/{chat_id}", response_model=ChatDetailOut)
async def get_chat(
    org_id: str,
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = ChatService(db)
    chat = await svc.get_chat(chat_id, org_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return ChatDetailOut(
        id=str(chat.id),
        title=chat.title,
        created_at=chat.created_at.isoformat(),
        updated_at=chat.updated_at.isoformat(),
        messages=[
            MessageOut(
                id=str(m.id),
                role=m.role,
                content=m.content,
                created_at=m.created_at.isoformat(),
            )
            for m in chat.messages
        ],
    )


@chat_router.post("/stream")
async def stream_message(
    org_id: str,
    body: SendMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = ChatService(db)

    # Get org's active KB collection
    collection_name = await svc.get_active_collection(org_id)

    # Create or fetch chat
    if body.chat_id:
        chat = await svc.get_chat(body.chat_id, org_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
    else:
        chat = await svc.create_chat(org_id, str(current_user.id))

    # Save user message
    user_msg = await svc.add_message(str(chat.id), "user", body.message)
    await db.commit()

    chat_id_str = str(chat.id)
    user_msg_id_str = str(user_msg.id)

    # Build conversation history from existing messages
    existing_chat = await svc.get_chat(chat_id_str, org_id)
    history = []
    if existing_chat and existing_chat.messages:
        for m in existing_chat.messages:
            if str(m.id) == user_msg_id_str:
                continue
            history.append(
                ConversationMessage(
                    role=m.role,
                    content=m.content,
                    timestamp=m.created_at.isoformat(),
                )
            )

    needs_title = not chat.title

    async def _generate_title(question: str, answer_text: str) -> str:
        """Use LLM to generate a short 3-5 word chat title."""
        try:
            llm = get_managed_llm(temperature=0)
            resp = await llm.ainvoke(
                f"Generate a concise 3-5 word title for this conversation. "
                f"Return ONLY the title, no quotes, no punctuation at the end.\n\n"
                f"User: {question}\nAssistant: {answer_text[:300]}"
            )
            title = resp.content.strip().strip('"').strip("'")
            return title[:100] if title else question[:80]
        except Exception:
            return question[:80]

    async def event_stream():
        yield _sse(
            "chat_init",
            {
                "chat_id": chat_id_str,
                "message_id": user_msg_id_str,
            },
        )

        payload: Dict[str, Any] = {
            "input_query": body.message,
            "conversation_history": history,
            "pending_clarification": False,
        }
        if collection_name:
            payload["collection_name"] = collection_name

        streamed_answer = ""
        final_answer = ""
        is_blocked = False
        blocked_reason = ""
        needs_clarification = False
        clarification_message = None
        answer_streaming = False
        answer_done = False

        try:
            # Dual stream: "updates" for node events, "messages" for LLM tokens
            async for mode, chunk in _graph.astream(
                payload, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    for node_name, state_update in chunk.items():
                        # Only show status for nodes before the answer streams
                        if not answer_streaming:
                            status_msg = NODE_STATUS_MAP.get(
                                node_name, f"Processing {node_name}..."
                            )
                            yield _sse(
                                "status", {"node": node_name, "message": status_msg}
                            )

                        if isinstance(state_update, dict):
                            if state_update.get("is_blocked"):
                                is_blocked = True
                                blocked_reason = state_update.get("blocked_reason", "")
                            if state_update.get("needs_clarification"):
                                needs_clarification = True
                                clarification_message = state_update.get(
                                    "clarification_message", ""
                                )
                            # Capture the final answer after output_guardrail
                            if (
                                "generated_answer" in state_update
                                and state_update["generated_answer"]
                            ):
                                final_answer = state_update["generated_answer"]

                elif mode == "messages":
                    msg_chunk, metadata = chunk
                    node = metadata.get("langgraph_node", "")
                    if (
                        node == "generate_answer"
                        and isinstance(msg_chunk, AIMessageChunk)
                        and msg_chunk.content
                    ):
                        if not answer_streaming:
                            answer_streaming = True
                            yield _sse(
                                "status",
                                {
                                    "node": "generate_answer",
                                    "message": "Generating answer...",
                                },
                            )
                        token = (
                            msg_chunk.content
                            if isinstance(msg_chunk.content, str)
                            else str(msg_chunk.content)
                        )
                        streamed_answer += token
                        yield _sse("answer_chunk", {"chunk": token})

            # Use the final post-guardrail answer
            answer = final_answer or streamed_answer

            # If guardrail modified the answer, replace what was streamed
            if answer and streamed_answer and answer != streamed_answer:
                yield _sse("answer_replace", {"content": answer})

            # If blocked/clarification, send that as the answer
            if is_blocked:
                answer = (
                    blocked_reason or "Your query was blocked by our safety filters."
                )
                yield _sse("answer_replace", {"content": answer})
            elif needs_clarification:
                answer = (
                    clarification_message or "Could you please provide more details?"
                )
                yield _sse("answer_replace", {"content": answer})
            elif not answer:
                answer = "I wasn't able to generate a response. Please try again."
                yield _sse("answer_chunk", {"chunk": answer})

            yield _sse("answer_done", {})

            # Generate title for new chats using LLM
            chat_title = chat.title
            if needs_title:
                chat_title = await _generate_title(body.message, answer)
                yield _sse("title", {"title": chat_title})

            # Save assistant message using a fresh session
            async with async_session() as save_db:
                save_db.add(
                    Message(
                        id=uuid.uuid4(),
                        chat_id=uuid.UUID(chat_id_str),
                        role="assistant",
                        content=answer,
                    )
                )
                if needs_title:
                    await save_db.execute(
                        Chat.__table__.update()
                        .where(Chat.id == uuid.UUID(chat_id_str))
                        .values(title=chat_title)
                    )
                await save_db.commit()

            yield _sse(
                "complete",
                {
                    "chat_id": chat_id_str,
                    "message_id": user_msg_id_str,
                    "title": chat_title,
                },
            )

        except Exception as exc:
            logger.exception("Chat stream error")
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@chat_router.patch("/{chat_id}", response_model=ChatOut)
async def rename_chat(
    org_id: str,
    chat_id: str,
    body: RenameChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = ChatService(db)
    chat = await svc.get_chat(chat_id, org_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    await svc.update_chat_title(chat_id, body.title)
    await db.commit()
    updated = await svc.get_chat(chat_id, org_id)
    return ChatOut(
        id=str(updated.id),
        title=updated.title,
        created_at=updated.created_at.isoformat(),
        updated_at=updated.updated_at.isoformat(),
    )


@chat_router.delete("/{chat_id}")
async def delete_chat(
    org_id: str,
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = ChatService(db)
    deleted = await svc.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")
    await db.commit()
    return {"message": "Chat deleted"}