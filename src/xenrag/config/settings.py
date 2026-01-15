import os
from dotenv import load_dotenv

load_dotenv()
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