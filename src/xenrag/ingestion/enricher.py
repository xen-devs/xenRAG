"""
Enrichment Module
Adds sentiment analysis and metadata to segments.
"""

import re
from typing import Dict, Any, List
from dataclasses import dataclass


# Sentiment lexicon (simplified)
POSITIVE_WORDS = {
    "great", "good", "excellent", "amazing", "awesome", "love", "perfect",
    "best", "fantastic", "wonderful", "happy", "satisfied", "recommend",
    "easy", "fast", "works", "nice", "beautiful", "solid", "reliable"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "worst", "hate", "poor", "broken",
    "slow", "difficult", "problem", "issue", "defective", "disappointed",
    "waste", "useless", "frustrating", "annoying", "junk", "cheap", "fail"
}

INTENSIFIERS = {"very", "really", "extremely", "absolutely", "totally", "so"}
NEGATORS = {"not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't"}


@dataclass
class SentimentResult:
    sentiment: str  # positive, negative, neutral
    score: float    # -1.0 to 1.0
    confidence: float


def analyze_sentiment(text: str) -> SentimentResult:
    """
    Rule-based sentiment analysis.
    Returns sentiment classification with score.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if not words:
        return SentimentResult("neutral", 0.0, 0.5)
    
    positive_count = 0
    negative_count = 0
    has_negator = False
    has_intensifier = False
    
    for i, word in enumerate(words):
        # Check for negator in previous 3 words
        if word in NEGATORS:
            has_negator = True
        if word in INTENSIFIERS:
            has_intensifier = True
        
        # Look back for negator
        negated = any(words[max(0, i-3):i].count(neg) for neg in NEGATORS)
        
        if word in POSITIVE_WORDS:
            if negated:
                negative_count += 1
            else:
                positive_count += 1
        elif word in NEGATIVE_WORDS:
            if negated:
                positive_count += 1
            else:
                negative_count += 1
    
    # Apply intensifier boost
    multiplier = 1.5 if has_intensifier else 1.0
    positive_count *= multiplier
    negative_count *= multiplier
    
    total = positive_count + negative_count
    
    if total == 0:
        return SentimentResult("neutral", 0.0, 0.5)
    
    score = (positive_count - negative_count) / max(total, 1)
    score = max(-1.0, min(1.0, score))
    
    if score > 0.2:
        sentiment = "positive"
    elif score < -0.2:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    confidence = min(0.9, 0.5 + (total * 0.05))
    
    return SentimentResult(sentiment, score, confidence)


def enrich_segment(segment_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich a segment with sentiment and additional metadata.
    """
    text = segment_dict.get("text", "")
    
    sentiment_result = analyze_sentiment(text)
    
    enriched = segment_dict.copy()
    enriched["sentiment"] = sentiment_result.sentiment
    enriched["sentiment_score"] = sentiment_result.score
    enriched["sentiment_confidence"] = sentiment_result.confidence
    
    # Add word count
    enriched["word_count"] = len(text.split())
    
    return enriched


def enrich_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich multiple segments."""
    return [enrich_segment(s) for s in segments]
