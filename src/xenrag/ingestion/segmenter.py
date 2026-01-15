"""
Semantic Segmentation Module
Splits reviews into aspect-level chunks.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


# Aspect keywords for rule-based detection
ASPECT_KEYWORDS = {
    "battery": ["battery", "batteries", "charge", "charging", "power", "drain", "dies"],
    "shipping": ["shipping", "delivery", "arrived", "package", "shipped", "box", "packaging"],
    "price": ["price", "cost", "expensive", "cheap", "worth", "money", "value", "afford"],
    "quality": ["quality", "build", "material", "durable", "sturdy", "flimsy", "broke", "broken"],
    "performance": ["fast", "slow", "speed", "performance", "responsive", "lag", "loading"],
    "display": ["screen", "display", "resolution", "picture", "visual", "color", "bright"],
    "audio": ["sound", "audio", "speaker", "volume", "loud", "quiet", "noise"],
    "setup": ["setup", "install", "installation", "configure", "easy", "difficult"],
    "customer_service": ["support", "service", "customer", "help", "response", "refund", "return"],
    "remote": ["remote", "controller", "buttons", "click", "press"],
}


@dataclass
class Segment:
    text: str
    aspect: str
    confidence: float
    start_pos: int
    end_pos: int


def detect_aspect(text: str) -> tuple[str, float]:
    """
    Detect the primary aspect of a text segment.
    Returns (aspect, confidence).
    """
    text_lower = text.lower()
    aspect_scores = {}
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            aspect_scores[aspect] = score
    
    if not aspect_scores:
        return ("general", 0.5)
    
    best_aspect = max(aspect_scores, key=aspect_scores.get)
    max_score = aspect_scores[best_aspect]
    confidence = min(0.9, 0.5 + (max_score * 0.1))
    
    return (best_aspect, confidence)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Handle common abbreviations
    text = re.sub(r'\.\.\.', '<ELLIPSIS>', text)
    text = re.sub(r'Mr\.', 'Mr<DOT>', text)
    text = re.sub(r'Mrs\.', 'Mrs<DOT>', text)
    text = re.sub(r'Dr\.', 'Dr<DOT>', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore
    sentences = [s.replace('<ELLIPSIS>', '...').replace('<DOT>', '.') for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]


def segment_review(text: str, min_segment_length: int = 10) -> List[Segment]:
    """
    Segment a review into aspect-tagged chunks.
    Groups consecutive sentences about the same aspect.
    """
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    segments = []
    current_sentences = []
    current_aspect = None
    current_start = 0
    pos = 0
    
    for sentence in sentences:
        aspect, confidence = detect_aspect(sentence)
        
        if current_aspect is None:
            current_aspect = aspect
            current_sentences = [sentence]
            current_start = pos
        elif aspect == current_aspect or aspect == "general":
            current_sentences.append(sentence)
        else:
            # New aspect - flush current segment
            segment_text = " ".join(current_sentences)
            if len(segment_text) >= min_segment_length:
                _, conf = detect_aspect(segment_text)
                segments.append(Segment(
                    text=segment_text,
                    aspect=current_aspect,
                    confidence=conf,
                    start_pos=current_start,
                    end_pos=pos
                ))
            current_sentences = [sentence]
            current_aspect = aspect
            current_start = pos
        
        pos += len(sentence) + 1
    
    # Flush final segment
    if current_sentences:
        segment_text = " ".join(current_sentences)
        if len(segment_text) >= min_segment_length:
            _, conf = detect_aspect(segment_text)
            segments.append(Segment(
                text=segment_text,
                aspect=current_aspect or "general",
                confidence=conf,
                start_pos=current_start,
                end_pos=pos
            ))
    
    # If no segments were created, return the whole text as one segment
    if not segments and text.strip():
        aspect, conf = detect_aspect(text)
        segments.append(Segment(
            text=text.strip(),
            aspect=aspect,
            confidence=conf,
            start_pos=0,
            end_pos=len(text)
        ))
    
    return segments


def segment_to_dict(segment: Segment) -> Dict[str, Any]:
    """Convert segment to dictionary for storage."""
    return {
        "text": segment.text,
        "aspect": segment.aspect,
        "aspect_confidence": segment.confidence,
    }
