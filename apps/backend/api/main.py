from contextlib import asynccontextmanager
from typing import Dict

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth.router import auth_router, orgs_router
from api.chat.router import chat_router
from api.core.config import CORS_ORIGINS
from api.db.models import (  # noqa: F401 — ensure models are registered
    User,
    Organization,
    UserOrganization,
    KnowledgeBaseDataset,
    KnowledgeBaseUpload,
    Chat,
    Message,
)
from api.knowledge_base.router import kb_router
from ai_core.graph.graph import setup_checkpointer


@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_checkpointer()
    yield


app = FastAPI(title="xenRAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/v1")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> Dict[str, str]:
    return {"status": "ready"}


router.include_router(auth_router)
router.include_router(orgs_router)
router.include_router(kb_router)
router.include_router(chat_router)
app.include_router(router)
