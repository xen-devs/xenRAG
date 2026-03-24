"""
Entity Extraction Module
Extracts entities and relationships for Knowledge Graph.
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


# Entity patterns
PRODUCT_PATTERNS = [
    r'\b(fire\s*stick|firestick)\b',
    r'\b(roku)\b',
    r'\b(chromecast)\b',
    r'\b(apple\s*tv)\b',
    r'\b(echo|alexa)\b',
    r'\b(kindle)\b',
    r'\b(remote)\b',
]

FEATURE_KEYWORDS = {
    "battery": ["battery", "batteries", "power", "charge"],
    "wifi": ["wifi", "wi-fi", "internet", "connection", "network"],
    "streaming": ["streaming", "stream", "netflix", "hulu", "prime", "youtube"],
    "voice_control": ["voice", "alexa", "command", "speech"],
    "display": ["screen", "display", "picture", "resolution", "4k", "hd"],
    "remote": ["remote", "controller", "buttons"],
    "speed": ["speed", "fast", "slow", "loading", "performance"],
    "setup": ["setup", "install", "installation", "configure"],
    "apps": ["app", "apps", "application", "store"],
}


@dataclass
class Entity:
    text: str
    entity_type: str
    start_pos: int
    end_pos: int


@dataclass
class Relationship:
    source_type: str
    source_text: str
    relation: str
    target_type: str
    target_text: str


def extract_entities(text: str) -> List[Entity]:
    """Extract named entities from text."""
    entities = []
    text_lower = text.lower()
    
    # Extract product mentions
    for pattern in PRODUCT_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            entities.append(Entity(
                text=match.group(),
                entity_type="Product",
                start_pos=match.start(),
                end_pos=match.end()
            ))
    
    # Extract feature mentions
    for feature, keywords in FEATURE_KEYWORDS.items():
        for keyword in keywords:
            for match in re.finditer(rf'\b{keyword}\b', text_lower):
                entities.append(Entity(
                    text=feature,
                    entity_type="Feature",
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
                break  # One match per feature type is enough
    
    return entities


def extract_relationships(
    text: str, 
    entities: List[Entity],
    sentiment: str = "neutral"
) -> List[Relationship]:
    """Extract relationships between entities."""
    relationships = []
    
    # Create Review -> Feature relationships based on mentions
    features_mentioned = set()
    products_mentioned = set()
    
    for entity in entities:
        if entity.entity_type == "Feature":
            features_mentioned.add(entity.text)
        elif entity.entity_type == "Product":
            products_mentioned.add(entity.text)
    
    # Relationship: Review mentions Feature with sentiment
    for feature in features_mentioned:
        relation = f"MENTIONS_{sentiment.upper()}"
        relationships.append(Relationship(
            source_type="Review",
            source_text="review",
            relation=relation,
            target_type="Feature",
            target_text=feature
        ))
    
    # Relationship: Review mentions Product
    for product in products_mentioned:
        relationships.append(Relationship(
            source_type="Review",
            source_text="review",
            relation="REVIEWS",
            target_type="Product",
            target_text=product
        ))
    
    return relationships


def extract_for_graph(segment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entities and relationships from a segment for Neo4j.
    Returns enhanced segment with graph data.
    """
    text = segment.get("text", "")
    sentiment = segment.get("sentiment", "neutral")
    
    entities = extract_entities(text)
    relationships = extract_relationships(text, entities, sentiment)
    
    enhanced = segment.copy()
    enhanced["entities"] = [
        {"text": e.text, "type": e.entity_type}
        for e in entities
    ]
    enhanced["relationships"] = [
        {
            "source_type": r.source_type,
            "relation": r.relation,
            "target_type": r.target_type,
            "target_text": r.target_text
        }
        for r in relationships
    ]
    
    # Extract feature list for easy querying
    enhanced["features"] = list(set(e.text for e in entities if e.entity_type == "Feature"))
    enhanced["products"] = list(set(e.text for e in entities if e.entity_type == "Product"))
    
    return enhanced
