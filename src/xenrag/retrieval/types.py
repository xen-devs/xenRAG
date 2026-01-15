from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class RetrievalItem:
    """Represents a single retrieved document/node."""
    id: str
    content: str
    source: str  # "qdrant" or "neo4j"
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchContext:
    """Context for a retrieval request."""
    query: str
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 5
    strategy: str = "HYBRID"  # "VECTOR_ONLY", "HYBRID"

@dataclass
class RetrievalResponse:
    """The final response from the RAG Tool."""
    items: List[RetrievalItem]
    total_found: int
    strategy_used: str
    debug_info: Dict[str, Any] = field(default_factory=dict)
