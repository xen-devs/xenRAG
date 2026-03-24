"""
Output Rail: Validates LLM responses before returning to user.
Checks for hallucinations, toxic content, and enforces quality.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Toxic patterns in output
TOXIC_OUTPUT_PATTERNS = [
    r"\b(kill|murder|hurt|harm|attack)\s+(yourself|someone|people)\b",
    r"\b(suicide|self.?harm)\b",
    r"\bI\s+(hate|despise)\s+(you|users|humans)\b",
]

# Confidence threshold patterns
LOW_CONFIDENCE_PHRASES = [
    "I'm not sure",
    "I don't know",
    "I cannot find",
    "no information",
    "unable to determine",
    "insufficient data",
]

# Hallucination indicators (claims without evidence markers)
UNSUPPORTED_CLAIM_PATTERNS = [
    r"studies\s+show\s+that",
    r"research\s+(proves?|indicates?)\s+that",
    r"according\s+to\s+experts?",
    r"it\s+is\s+(well\s+)?known\s+that",
    r"scientists?\s+(agree|confirm)\s+that",
]


@dataclass
class OutputRailResult:
    is_safe: bool
    modified_response: str
    blocked_reason: str = ""
    warnings: List[str] = None
    confidence_score: float = 1.0
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def detect_toxic_output(text: str) -> bool:
    """Check if output contains toxic content."""
    text_lower = text.lower()
    
    for pattern in TOXIC_OUTPUT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def check_low_confidence(text: str) -> float:
    """Estimate confidence based on hedging language."""
    text_lower = text.lower()
    hedging_count = 0
    
    for phrase in LOW_CONFIDENCE_PHRASES:
        if phrase.lower() in text_lower:
            hedging_count += 1
    
    # More hedging = lower confidence
    if hedging_count >= 3:
        return 0.3
    elif hedging_count >= 2:
        return 0.5
    elif hedging_count >= 1:
        return 0.7
    
    return 0.9


def check_hallucination_risk(
    response: str,
    source_docs: Optional[List[str]] = None
) -> List[str]:
    """
    Check for potential hallucination indicators.
    Returns list of warnings.
    """
    warnings = []
    
    # Check for unsupported claim patterns
    for pattern in UNSUPPORTED_CLAIM_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            warnings.append("Response contains claims that may not be from source documents")
            break
    
    # If we have source docs, check if response contains info not in sources
    if source_docs:
        # Extract key claims from response (simplified)
        response_sentences = re.split(r'[.!?]', response)
        source_text = " ".join(source_docs).lower()
        
        for sentence in response_sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Check for specific numbers/dates not in sources
            numbers_in_response = re.findall(r'\b\d+(?:\.\d+)?%?\b', sentence)
            for num in numbers_in_response:
                if num not in source_text and len(num) > 1:
                    warnings.append(f"Statistic '{num}' may not be from source documents")
                    break
    
    return warnings


def sanitize_output(text: str) -> str:
    """Remove or fix problematic content in output."""
    sanitized = text
    
    # Remove any system prompt leakage
    sanitized = re.sub(r"\[System\].*?\[/System\]", "", sanitized, flags=re.DOTALL)
    sanitized = re.sub(r"<\|[^|]+\|>", "", sanitized)
    
    # Remove repeated phrases (sign of model issues)
    words = sanitized.split()
    if len(words) > 10:
        deduplicated = []
        prev_window = []
        for word in words:
            prev_window.append(word)
            if len(prev_window) > 5:
                prev_window.pop(0)
            
            # Check for exact repetition
            window_str = " ".join(prev_window[-3:])
            if deduplicated and " ".join(deduplicated[-3:]) == window_str:
                continue
            deduplicated.append(word)
        
        if len(deduplicated) < len(words):
            sanitized = " ".join(deduplicated)
    
    return sanitized.strip()


def add_disclaimer_if_needed(
    response: str,
    confidence: float,
    warnings: List[str]
) -> str:
    """Add appropriate disclaimers based on confidence and warnings."""
    if confidence < 0.5 or len(warnings) > 1:
        disclaimer = "\n\n*Note: This response is based on limited information from the available reviews.*"
        if disclaimer not in response:
            response = response + disclaimer
    
    return response


def validate_output(
    response: str,
    source_docs: Optional[List[str]] = None,
    add_disclaimers: bool = True
) -> OutputRailResult:
    """
    Main output validation function.
    """
    if not response or not response.strip():
        return OutputRailResult(
            is_safe=False,
            modified_response="I couldn't generate a response. Please try rephrasing your question.",
            blocked_reason="Empty response"
        )
    
    # Check for toxic content
    if detect_toxic_output(response):
        logger.warning("Toxic content detected in output")
        return OutputRailResult(
            is_safe=False,
            modified_response="I apologize, but I cannot provide that response. Please ask about customer reviews.",
            blocked_reason="Toxic content in response"
        )
    
    # Check confidence
    confidence = check_low_confidence(response)
    
    # Check for hallucination risks
    warnings = check_hallucination_risk(response, source_docs)
    
    # Sanitize output
    sanitized = sanitize_output(response)
    
    # Add disclaimers if needed
    if add_disclaimers:
        sanitized = add_disclaimer_if_needed(sanitized, confidence, warnings)
    
    return OutputRailResult(
        is_safe=True,
        modified_response=sanitized,
        warnings=warnings,
        confidence_score=confidence
    )
