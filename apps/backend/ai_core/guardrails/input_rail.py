"""
Input Rail: Validates user input before processing.
Detects jailbreak attempts, toxic content, and PII.
"""

import re
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Jailbreak patterns - common prompt injection attempts
JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions?|programming)",
    r"forget\s+(everything|all|your)\s+(you\s+)?(know|learned|instructions?)",
    r"you\s+are\s+now\s+(a|an)\s+\w+\s+(that|who|which)",
    r"pretend\s+(you\s+are|to\s+be)\s+(a|an)?\s*\w+",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a|an)?\s*\w+",
    r"simulate\s+(being|a|an)\s+\w+",
    r"roleplay\s+as\s+(a|an)?\s*\w+",
    r"bypass\s+(your\s+)?(safety|security|restrictions?|guidelines?)",
    r"override\s+(your\s+)?(safety|security|programming)",
    r"sudo\s+mode",
    r"developer\s+mode",
    r"jailbreak",
    r"DAN\s+mode",
    r"\[system\]",
    r"\[INST\]",
    r"<\|im_start\|>",
]

# Toxic/harmful content patterns
TOXIC_PATTERNS = [
    r"\b(kill|murder|hurt|harm|attack)\s+(yourself|myself|someone|people|them)\b",
    r"\b(suicide|self.?harm)\b",
    r"\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive)\b",
    r"\b(hate|racist|sexist)\s+(speech|content)\b",
]

# PII patterns for detection
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
    "credit_card": r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b",
}


@dataclass
class InputRailResult:
    is_safe: bool
    blocked_reason: str = ""
    risk_type: str = ""
    sanitized_query: str = ""
    pii_detected: List[str] = None
    
    def __post_init__(self):
        if self.pii_detected is None:
            self.pii_detected = []


def detect_jailbreak(text: str) -> Tuple[bool, str]:
    """Detect jailbreak/prompt injection attempts."""
    text_lower = text.lower()
    
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True, f"Matched jailbreak pattern: {pattern[:30]}..."
    
    return False, ""


def detect_toxic_content(text: str) -> Tuple[bool, str]:
    """Detect harmful or toxic content."""
    text_lower = text.lower()
    
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True, "Harmful content detected"
    
    return False, ""


def detect_pii(text: str) -> List[str]:
    """Detect potential PII in text."""
    detected = []
    
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            detected.append(pii_type)
    
    return detected


def sanitize_query(text: str) -> str:
    """Sanitize query by removing or masking sensitive content."""
    sanitized = text
    
    # Mask PII
    for pii_type, pattern in PII_PATTERNS.items():
        sanitized = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", sanitized)
    
    # Remove potential injection markers
    sanitized = re.sub(r"<\|[^|]+\|>", "", sanitized)
    sanitized = re.sub(r"\[(?:system|INST|/INST)\]", "", sanitized)
    
    return sanitized.strip()


def validate_input(query: str) -> InputRailResult:
    """
    Main input validation function.
    Returns InputRailResult with safety status and details.
    """
    if not query or not query.strip():
        return InputRailResult(
            is_safe=False,
            blocked_reason="Empty query provided",
            risk_type="invalid_input"
        )
    
    # Check for jailbreak attempts
    is_jailbreak, jailbreak_reason = detect_jailbreak(query)
    if is_jailbreak:
        logger.warning(f"Jailbreak attempt detected: {jailbreak_reason}")
        return InputRailResult(
            is_safe=False,
            blocked_reason="I can't process requests that try to override my guidelines. Please ask about customer reviews.",
            risk_type="jailbreak",
            sanitized_query=""
        )
    
    # Check for toxic content
    is_toxic, toxic_reason = detect_toxic_content(query)
    if is_toxic:
        logger.warning(f"Toxic content detected in query")
        return InputRailResult(
            is_safe=False,
            blocked_reason="I can't help with that request. Please ask about customer reviews and product feedback.",
            risk_type="toxic_content",
            sanitized_query=""
        )
    
    # Detect PII (warn but don't block)
    pii_found = detect_pii(query)
    if pii_found:
        logger.info(f"PII detected in query: {pii_found}")
    
    # Sanitize the query
    sanitized = sanitize_query(query)
    
    return InputRailResult(
        is_safe=True,
        sanitized_query=sanitized,
        pii_detected=pii_found
    )
