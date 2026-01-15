import logging
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from xenrag.retrieval.interfaces import VectorStore
from xenrag.retrieval.types import RetrievalItem, SearchContext
from xenrag.config import settings
from xenrag.retrieval.embedding import Embedder

logger = logging.getLogger(__name__)

class QdrantVectorStore(VectorStore):
    """
    Production-grade Qdrant Adapter with Logging, Error Handling, and Real Embeddings.
    """
    def __init__(self, url: Optional[str] = None, collection_name: Optional[str] = None):
        self.url = url or settings.QDRANT_URL
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        
        logger.info(f"Connecting to Qdrant at {self.url} [Collection: {self.collection_name}]")
        
        try:
            self.client = QdrantClient(url=self.url)
            # Initialize Embedder
            self.embedder = Embedder() 
            self._ensure_collection()
            logger.info("Qdrant connection successful.")
        except Exception as e:
            logger.critical(f"Failed to connect to Qdrant: {e}")
            raise

    def _ensure_collection(self):
        """Check if collection exists, create if not."""
        try:
            # Determine embedding dimension dynamically
            test_embedding = self.embedder.embed_query("test")
            self.embedding_dim = len(test_embedding)
            logger.info(f"Detected embedding dimension: {self.embedding_dim}")

            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists:
                # Check if dimension matches
                collection_info = self.client.get_collection(self.collection_name)
                current_dim = collection_info.config.params.vectors.size
                
                if current_dim != self.embedding_dim:
                    logger.warning(f"Collection '{self.collection_name}' dimension mismatch! Existing: {current_dim}, New: {self.embedding_dim}. Recreating...")
                    self.client.delete_collection(self.collection_name)
                    exists = False
            
            if not exists:
                logger.warning(f"Collection '{self.collection_name}' not found or deleted. Creating new...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE)
                )
                logger.info(f"Collection '{self.collection_name}' created with dimension {self.embedding_dim}.")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def search(self, context: SearchContext) -> List[RetrievalItem]:
        """
        Execute vector search.
        """
        logger.debug(f"Searching Qdrant for query: '{context.query}'")
        
        try:
            query_vector = self.embedder.embed_query(context.query)
            
            q_filter = None
            if context.filters:
                must_conditions = []
                for key, value in context.filters.items():
                    must_conditions.append(
                        models.FieldCondition(
                            key=key, 
                            match=models.MatchValue(value=value)
                        )
                    )
                q_filter = models.Filter(must=must_conditions)

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector, 
                query_filter=q_filter,
                limit=context.limit
            )
            results = response.points
            
            logger.info(f"Qdrant returned {len(results)} hits.")

            return [
                RetrievalItem(
                    id=str(hit.id),
                    content=hit.payload.get("text", ""),
                    source="qdrant",
                    score=hit.score,
                    metadata=hit.payload
                )
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Ingest documents. 
        """
        logger.info(f"Ingesting {len(documents)} documents to Qdrant.")
        try:
            points = []
            
            texts_to_embed = [doc.get("text", "") for doc in documents]
            embeddings = self.embedder.embed_documents(texts_to_embed)
            
            for i, doc in enumerate(documents):
                payload = doc.copy()
                if "embedding" in payload:
                    del payload["embedding"]
                
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i],
                    payload=payload
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info("Ingestion complete.")
        except Exception as e:
            logger.error(f"Qdrant ingestion failed: {e}")
            raise
