from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.auth.router import auth_router, orgs_router
from api.core.config import CORS_ORIGINS
from api.db.models import User, Organization, UserOrganization  # noqa: F401 — ensure models are registered
from api.schemas import ChatRequest, ChatResponse
from ai_core.graph.graph import build_graph
from ai_core.graph.state import ConversationMessage


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="xenRAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = build_graph()
router = APIRouter(prefix="/api/v1")


def _model_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_model_to_dict(item) for item in value]
    return value


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> Dict[str, str]:
    return {"status": "ready"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.input_query.strip():
        raise HTTPException(status_code=400, detail="input_query cannot be empty")

    history = [
        ConversationMessage(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp or datetime.now().isoformat(),
        )
        for msg in request.conversation_history
    ]

    payload = {
        "input_query": request.input_query,
        "conversation_history": history,
        "pending_clarification": request.pending_clarification,
    }

    try:
        result = await _graph.ainvoke(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {exc}") from exc

    return ChatResponse(
        intent=_model_to_dict(result.get("intent")),
        emotion=_model_to_dict(result.get("emotion")),
        is_blocked=bool(result.get("is_blocked", False)),
        blocked_reason=result.get("blocked_reason", ""),
        needs_clarification=bool(result.get("needs_clarification", False)),
        clarification_message=result.get("clarification_message"),
        clarification_reason=result.get("clarification_reason"),
        generated_answer=result.get("generated_answer"),
        explanations=_model_to_dict(result.get("explanations", [])),
        private_reasoning=_model_to_dict(result.get("private_reasoning", [])),
    )


router.include_router(auth_router)
router.include_router(orgs_router)
app.include_router(router)
