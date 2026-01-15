"""
Query Node: Decides retrieval strategy and performs retrieval.
"""

from typing import Dict, Any, List
from xenrag.graph.state import GraphState, ReasoningRecord, RetrievalContext, RetrievalItem
from xenrag.retrieval.engine import RagEngine
from xenrag.config import settings

async def query_node(state: GraphState) -> Dict[str, Any]:
    """
    Decides the retrieval strategy (Vector vs Hybrid) based on intent
    and executes retrieval using the real RagTool.
    """
    print("--- QUERY NODE ---")
    intent = state.intent
    
    strategy = "HYBRID"
    reasoning_summary = "Using default HYBRID strategy."
    
    if intent:
        if intent.type in ["specific_question", "feature_request"]:
            strategy = "VECTOR_ONLY"
            reasoning_summary = f"Intent '{intent.type}' detected. Using VECTOR_ONLY for precision."
        elif intent.type in ["complaint_analysis", "summary_request"]:
            strategy = "HYBRID"
            reasoning_summary = f"Intent '{intent.type}' detected. Using HYBRID for coverage."
            
    print(f"Selected Strategy: {strategy}")

    engine = RagEngine()
    
    try:
        query_text = state.input_query
        
        retrieval_limit = settings.RAG_RETRIEVAL_LIMIT
        
        response = await engine.search(
            query=query_text,
            strategy=strategy,
            limit=retrieval_limit
        )
        
        def convert_items(items):
            return [
                RetrievalItem(
                    id=str(item.id),
                    content=item.content,
                    source=item.source,
                    score=item.score,
                    metadata={str(k): str(v) for k,v in item.metadata.items()}
                ) for item in items
            ]
        
        vector_items = [i for i in response.items if i.source == 'qdrant']
        graph_items = [i for i in response.items if i.source == 'neo4j']
        
        context = RetrievalContext(
            vector_results=convert_items(vector_items),
            kg_results=convert_items(graph_items),
            merged_results=convert_items(response.items),
            retrieval_confidence=0.8 # TODO: Compute from scores
        )
        
        return {
            "retrieval_context": context,
            "private_reasoning": [
                ReasoningRecord(
                    step="Query",
                    summary=f"{reasoning_summary} Found {response.total_found} items.",
                    confidence=1.0
                )
            ]
        }
    except Exception as e:
        print(f"Query Node Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "retrieval_context": None,
             "private_reasoning": [
                ReasoningRecord(
                    step="Query",
                    summary=f"Retrieval failed: {e}",
                    confidence=0.0
                )
            ]
        }
