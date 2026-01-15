from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .types import RetrievalItem, SearchContext

class VectorStore(ABC):
    """Abstract interface for Vector Database operations."""
    
    @abstractmethod
    def search(self, context: SearchContext) -> List[RetrievalItem]:
        """Perform semantic search."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Ingest documents (optional for now)."""
        pass

class GraphStore(ABC):
    """Abstract interface for Knowledge Graph operations."""
    
    @abstractmethod
    def query(self, cypher_query: str, params: Dict[str, Any] = None) -> List[RetrievalItem]:
        """Execute a raw cypher query."""
        pass
    
    @abstractmethod
    def get_context(self, context: SearchContext) -> List[RetrievalItem]:
        """Retrieve relevant graph context for a search query."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], label: str = "Document") -> None:
        """
        Ingest documents as nodes.
        :param documents: List of dictionaries (properties).
        :param label: The node label to use (default: Document).
        """
        pass
