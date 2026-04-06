import os
from dotenv import load_dotenv

load_dotenv()
_raw_sync_url = os.getenv("DATABASE_URL_SYNC", "")
DATABASE_URL_SYNC = _raw_sync_url.replace("postgresql+psycopg2://", "postgresql://") if _raw_sync_url else ""

OLLAMA_URL = os.getenv("OLLAMA_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_EMBEDDING_MODEL = os.getenv("LLM_EMBEDDING_MODEL")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# RAG Settings
RAG_RETRIEVAL_LIMIT = int(os.getenv("RAG_RETRIEVAL_LIMIT", "10"))

# Conversation Settings
MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "5"))

# LLM Strategy: failover, round_robin, least_connections, random
LLM_STRATEGY = os.getenv("LLM_STRATEGY", "round_robin")

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")