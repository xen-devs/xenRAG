import asyncio
from typing import List, Dict, Any, Optional
from xenrag.retrieval.types import SearchContext, RetrievalResponse, RetrievalItem
from xenrag.retrieval.stores.qdrant import QdrantVectorStore
from xenrag.retrieval.stores.neo4j import Neo4jGraphStore

class RagEngine:
    """
    The main entry point for the Retrieval Augmentation Generation tool.
    Orchestrates Vector and Graph searches, merges results, and (optionally) reranks.
    """
    def __init__(self):
        self.vector_store = QdrantVectorStore()
        self.graph_store = Neo4jGraphStore()

    async def search(self, 
                     query: str, 
                     filters: Dict[str, Any] = None, 
                     strategy: str = "HYBRID",
                     limit: int = 5) -> RetrievalResponse:
        
        context = SearchContext(
            query=query, 
            filters=filters or {}, 
            limit=limit, 
            strategy=strategy
        )
        
        vector_results: List[RetrievalItem] = []
        graph_results: List[RetrievalItem] = []

        # Execute based on strategy
        if strategy == "VECTOR_ONLY":
            vector_results = self.vector_store.search(context)
            
        elif strategy == "HYBRID":
            # Run both in parallel (using asyncio.to_thread for blocking IO)
            # Since adapters are sync, we wrap them.
            vec_task = asyncio.to_thread(self.vector_store.search, context)
            graph_task = asyncio.to_thread(self.graph_store.get_context, context)
            
            vector_results, graph_results = await asyncio.gather(vec_task, graph_task)
            
        # Merge & Rerank
        merged = self._merge_results(vector_results, graph_results)
        
        return RetrievalResponse(
            items=merged,
            total_found=len(merged),
            strategy_used=strategy,
            debug_info={
                "vector_count": len(vector_results),
                "graph_count": len(graph_results)
            }
        )

    def _merge_results(self, vector_hits: List[RetrievalItem], graph_hits: List[RetrievalItem]) -> List[RetrievalItem]:
        """
        Simple merge strategy: Interleave or prioritize Vector high confidence.
        For now: returned combined list sorted by score (if graph has pseudo-score).
        """
        combined = vector_hits + graph_hits
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined

    def close(self):
        """Cleanup resources."""
        if hasattr(self.graph_store, "close"):
            self.graph_store.close()
