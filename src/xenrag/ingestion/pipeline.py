"""
Ingestion Pipeline
Orchestrates the full ingestion flow: Normalize -> Segment -> Enrich -> Store
"""

import logging
from typing import Dict, Any, List, Generator
from dataclasses import dataclass

from xenrag.ingestion.normalizer import normalize_text, clean_for_embedding
from xenrag.ingestion.segmenter import segment_review, segment_to_dict
from xenrag.ingestion.enricher import enrich_segment
from xenrag.ingestion.entity_extractor import extract_for_graph
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    original_id: str
    segments: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def process_document(doc: Dict[str, Any]) -> ProcessedDocument:
    """
    Process a single document through the ingestion pipeline.
    
    Pipeline stages:
    1. Extract text and normalize
    2. Segment into aspect-level chunks
    3. Enrich with sentiment
    4. Extract entities for graph
    """
    # Get original document ID
    doc_id = doc.get("id", doc.get("asin", ""))
    
    # Extract text content
    text = doc.get("text") or doc.get("content") or doc.get("review") or doc.get("body", "")
    
    if not text:
        return ProcessedDocument(
            original_id=doc_id,
            segments=[],
            metadata={}
        )
    
    # Stage 1: Normalize
    normalized_text = normalize_text(text)
    
    # Stage 2: Segment
    segments = segment_review(normalized_text)
    segment_dicts = [segment_to_dict(s) for s in segments]
    
    # Stage 3 & 4: Enrich + Extract entities
    enriched_segments = []
    for seg in segment_dicts:
        # Add sentiment
        enriched = enrich_segment(seg)
        # Add entities and relationships
        enriched = extract_for_graph(enriched)
        
        # Assign unique ID for cross-linking
        enriched["id"] = str(uuid.uuid4())
        
        # Add parent document reference
        enriched["parent_id"] = doc_id
        # Copy relevant metadata from original doc
        for key in ["rating", "user_id", "asin", "parent_asin", "verified_purchase"]:
            if key in doc:
                enriched[key] = doc[key]
        
        enriched_segments.append(enriched)
    
    # Collect metadata
    metadata = {
        "original_text_length": len(text),
        "normalized_text_length": len(normalized_text),
        "segment_count": len(enriched_segments),
        "rating": doc.get("rating"),
    }
    
    return ProcessedDocument(
        original_id=doc_id,
        segments=enriched_segments,
        metadata=metadata
    )


def process_batch(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a batch of documents and return flattened segments.
    Each segment becomes a separate document for storage.
    """
    all_segments = []
    
    for doc in documents:
        try:
            processed = process_document(doc)
            all_segments.extend(processed.segments)
        except Exception as e:
            logger.warning(f"Failed to process document: {e}")
            # Fallback: store original with minimal processing
            text = doc.get("text") or doc.get("content") or doc.get("review") or ""
            all_segments.append({
                "text": normalize_text(text),
                "aspect": "general",
                "sentiment": "neutral",
                "parent_id": doc.get("id", ""),
            })
    
    return all_segments


def process_for_vector_store(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare segments for vector store.
    Cleans text for embedding and includes all metadata.
    """
    vector_docs = []
    
    for seg in segments:
        text = seg.get("text", "")
        cleaned_text = clean_for_embedding(text)
        
        if not cleaned_text:
            continue
        
        vector_doc = {
            "id": seg.get("id"),
            "text": cleaned_text,
            "aspect": seg.get("aspect", "general"),
            "sentiment": seg.get("sentiment", "neutral"),
            "sentiment_score": seg.get("sentiment_score", 0.0),
            "features": seg.get("features", []),
            "parent_id": seg.get("parent_id", ""),
            "rating": seg.get("rating"),
        }
        
        vector_docs.append(vector_doc)
    
    return vector_docs


def process_for_graph_store(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare segments for graph store with relationship data.
    """
    graph_docs = []
    
    for seg in segments:
        graph_doc = {
            "id": seg.get("id"),
            "text": seg.get("text", ""),
            "aspect": seg.get("aspect", "general"),
            "sentiment": seg.get("sentiment", "neutral"),
            "sentiment_score": seg.get("sentiment_score", 0.0),
            "features": ",".join(seg.get("features", [])),
            "products": ",".join(seg.get("products", [])),
            "parent_id": seg.get("parent_id", ""),
            "rating": seg.get("rating"),
            "relationships": seg.get("relationships", []),
        }
        
        graph_docs.append(graph_doc)
    
    return graph_docs
