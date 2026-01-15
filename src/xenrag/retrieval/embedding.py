import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from xenrag.config.settings import LLM_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class Embedder:
    """
    Wrapper for Embedding Generation using HuggingFace Transformers.
    Defaults to 'all-MiniLM-L6-v2' which is a great balance of speed/performance.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or LLM_EMBEDDING_MODEL
        logger.info(f"Initializing HuggingFace Embedder with model={self.model_name}")
        self._client = HuggingFaceEmbeddings(
            model_name=self.model_name,
            encode_kwargs={'normalize_embeddings': True}
        )

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query string."""
        try:
            return self._client.embed_query(text)
        except Exception as e:
            logger.error(f"Embedding generation failed for query: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        try:
            return self._client.embed_documents(texts)
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise
