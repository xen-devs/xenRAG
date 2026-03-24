"""
Retrieval Rail: Filters and validates retrieved context.
Removes low-quality results and PII from retrieved documents.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Minimum relevance score threshold
MIN_RELEVANCE_SCORE = 0.3

# PII patterns to redact from retrieved content
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
    "credit_card": r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b",
}

# Spam/low-quality content indicators
SPAM_PATTERNS = [
    r"(click\s+here|buy\s+now|limited\s+time\s+offer)",
    r"(free\s+gift|winner|congratulations\s+you\s+won)",
    r"https?://[^\s]+\s*https?://[^\s]+",  # Multiple URLs
    r"(.)\1{10,}",  # Repeated characters
]


@dataclass
class RetrievalRailResult:
    filtered_results: List[Dict[str, Any]]
    removed_count: int
    pii_redacted: bool
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def is_spam_content(text: str) -> bool:
    """Check if content appears to be spam."""
    text_lower = text.lower()
    
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def is_low_quality(text: str, min_length: int = 10) -> bool:
    """Check if content is too short or low quality."""
    if len(text.strip()) < min_length:
        return True
    
    # Check for mostly non-alphanumeric
    alpha_count = sum(1 for c in text if c.isalnum())
    if alpha_count < len(text) * 0.3:
        return True
    
    return False


def redact_pii(text: str) -> tuple[str, bool]:
    """Redact PII from text. Returns (redacted_text, was_modified)."""
    original = text
    redacted = text
    
    for pii_type, pattern in PII_PATTERNS.items():
        redacted = re.sub(pattern, f"[REDACTED]", redacted)
    
    return redacted, redacted != original


def filter_by_relevance(
    results: List[Dict[str, Any]],
    min_score: float = MIN_RELEVANCE_SCORE
) -> tuple[List[Dict[str, Any]], int]:
    """Filter results below minimum relevance score."""
    filtered = []
    removed = 0
    
    for result in results:
        score = result.get("score", 0.0)
        if score >= min_score:
            filtered.append(result)
        else:
            removed += 1
            logger.debug(f"Removed low-relevance result: score={score}")
    
    return filtered, removed


def filter_retrieval_results(
    results: List[Dict[str, Any]],
    min_relevance: float = MIN_RELEVANCE_SCORE,
    redact_pii_content: bool = True,
    remove_spam: bool = True
) -> RetrievalRailResult:
    """
    Main retrieval filtering function.
    
    Args:
        results: List of retrieval results with 'content'/'text' and 'score' fields
        min_relevance: Minimum relevance score threshold
        redact_pii_content: Whether to redact PII from content
        remove_spam: Whether to filter spam content
    """
    if not results:
        return RetrievalRailResult(
            filtered_results=[],
            removed_count=0,
            pii_redacted=False
        )
    
    filtered = []
    removed_count = 0
    pii_redacted = False
    warnings = []
    
    for result in results:
        # Get content field
        content = result.get("content") or result.get("text", "")
        score = result.get("score", 0.0)
        
        # Filter by relevance
        if score < min_relevance:
            removed_count += 1
            continue
        
        # Filter spam
        if remove_spam and is_spam_content(content):
            removed_count += 1
            warnings.append("Spam content filtered")
            continue
        
        # Filter low quality
        if is_low_quality(content):
            removed_count += 1
            continue
        
        # Redact PII
        result_copy = result.copy()
        if redact_pii_content:
            redacted_content, was_redacted = redact_pii(content)
            if was_redacted:
                pii_redacted = True
                result_copy["content"] = redacted_content
                if "text" in result_copy:
                    result_copy["text"] = redacted_content
        
        filtered.append(result_copy)
    
    if removed_count > 0:
        logger.info(f"Retrieval rail: filtered {removed_count} results")
    
    return RetrievalRailResult(
        filtered_results=filtered,
        removed_count=removed_count,
        pii_redacted=pii_redacted,
        warnings=warnings
    )
