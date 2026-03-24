"""
Topic Rail: Ensures conversations stay on-topic.
Detects off-topic queries and provides appropriate redirects.
"""

import re
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# On-topic keywords for customer review context
ON_TOPIC_KEYWORDS = [
    "review", "rating", "star", "product", "quality", "price", "value",
    "shipping", "delivery", "battery", "performance", "feature",
    "customer", "feedback", "complaint", "issue", "problem", "love",
    "hate", "recommend", "buy", "purchase", "return", "refund",
    "fire stick", "firestick", "roku", "streaming", "remote", "app",
    "setup", "install", "work", "broken", "defective", "excellent",
    "terrible", "amazing", "worst", "best", "opinion", "think", "say"
]

# Follow-up keywords
FOLLOW_UP_KEYWORDS = [
    "yeah", "yes", "yep", "yup", "sure", "ok", "okay",
    "no", "nope", "nah",
    "negative", "positive", "lowest", "highest", "most", "least",
    "bad", "good", "better", "worse", "first", "last",
    "one", "that", "this", "those", "these",
    "want", "need", "looking", "mean"
]

# Off-topic categories and their patterns
OFF_TOPIC_PATTERNS = {
    "politics": [
        r"\b(election|president|democrat|republican|politics|vote|congress)\b",
        r"\b(trump|biden|obama|liberal|conservative)\b",
    ],
    "medical": [
        r"\b(diagnose|diagnosis|symptom|treatment|medication|doctor|medical)\b",
        r"\b(disease|illness|cure|prescription|health\s+advice)\b",
    ],
    "legal": [
        r"\b(lawsuit|sue|attorney|lawyer|legal\s+advice|court)\b",
        r"\b(illegal|crime|criminal|arrest)\b",
    ],
    "financial": [
        r"\b(stock|invest|investment|bitcoin|crypto|trading)\b",
        r"\b(financial\s+advice|money\s+management|portfolio)\b",
    ],
    "personal": [
        r"\b(dating|relationship|love\s+life|marriage\s+advice)\b",
        r"\b(personal\s+problem|life\s+advice|emotional\s+support)\b",
    ],
    "weather": [
        r"\b(weather|forecast|temperature|rain|snow|sunny)\b",
    ],
    "general_knowledge": [
        r"\b(who\s+is|what\s+is\s+the\s+capital|history\s+of|when\s+was)\b",
        r"\b(how\s+old\s+is|where\s+is|famous\s+for)\b",
    ],
}

# Redirect messages for each off-topic category
REDIRECT_MESSAGES = {
    "politics": "I'm designed to help with customer reviews and product feedback. For political topics, please consult appropriate news sources.",
    "medical": "I can only help with product reviews. For medical questions, please consult a healthcare professional.",
    "legal": "I'm not able to provide legal advice. Please consult a qualified attorney for legal matters.",
    "financial": "I focus on product reviews, not financial advice. Please consult a financial advisor for investment questions.",
    "personal": "I'm here to help with product reviews and customer feedback. For personal matters, consider speaking with someone you trust.",
    "weather": "I specialize in customer reviews and product feedback, not weather forecasts. Please check a weather service.",
    "general_knowledge": "I'm specialized in analyzing customer reviews. For general knowledge questions, a search engine would be more helpful.",
    "default": "That question seems outside my area of expertise. I'm here to help with customer reviews and product feedback. How can I help you with that?"
}


@dataclass
class TopicRailResult:
    is_on_topic: bool
    off_topic_category: str = ""
    redirect_message: str = ""
    topic_confidence: float = 1.0


def calculate_topic_score(text: str, is_follow_up: bool = False) -> float:
    """Calculate how on-topic a query is based on keyword matching."""
    text_lower = text.lower()
    
    keyword_matches = sum(1 for kw in ON_TOPIC_KEYWORDS if kw in text_lower)
    
    if is_follow_up:
        follow_up_matches = sum(1 for kw in FOLLOW_UP_KEYWORDS if kw in text_lower)
        keyword_matches += follow_up_matches * 0.5
    
    max_expected = 5
    score = min(1.0, keyword_matches / max_expected)
    
    return score


def detect_off_topic(text: str) -> Tuple[bool, str]:
    """Detect if query is about an off-topic subject."""
    text_lower = text.lower()
    
    for category, patterns in OFF_TOPIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, category
    
    return False, ""


def is_greeting_or_meta(text: str) -> bool:
    """Check if query is a greeting or meta-question (allowed)."""
    greetings = [
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening))[\s!.,]*$",
        r"^(thanks?|thank\s+you)[\s!.,]*$",
        r"^(bye|goodbye|see\s+you)[\s!.,]*$",
        r"^(help|how\s+do\s+i|what\s+can\s+you).*$",
    ]
    
    text_lower = text.lower().strip()
    
    for pattern in greetings:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def validate_topic(
    query: str, 
    pending_clarification: bool = False,
    conversation_history: List = None
) -> TopicRailResult:
    """
    Main topic validation function.
    Determines if the query is on-topic for customer review analysis.
    
    Args:
        query: The user's query
        pending_clarification: Whether system is awaiting clarification response
        conversation_history: Recent conversation turns for context
    """
    if not query or not query.strip():
        return TopicRailResult(
            is_on_topic=False,
            redirect_message="Please ask a question about customer reviews or product feedback."
        )
    
    # Allow greetings and meta questions
    if is_greeting_or_meta(query):
        return TopicRailResult(is_on_topic=True, topic_confidence=1.0)
    
    # If responding to clarification, use relaxed validation
    if pending_clarification:
        logger.info("Processing clarification follow-up with relaxed validation")
        
        # Check if it relates to the clarification question
        if conversation_history:
            last_clarification = None
            for msg in reversed(conversation_history):
                role = msg.role if hasattr(msg, 'role') else msg.get('role')
                if role == 'clarification':
                    last_clarification = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    break
            
            if last_clarification:
                response_indicators = [
                    r"^(yes|yeah|yep|yup|no|nope|sure|okay|ok)\b",
                    r"\b(want|need|looking|mean|negative|positive|lowest|highest|worst|best)\b",
                    r"\b(one|first|last|most|least)\b"
                ]
                
                is_response = any(
                    re.search(p, query.lower()) for p in response_indicators
                )
                
                if is_response:
                    return TopicRailResult(
                        is_on_topic=True,
                        topic_confidence=0.7
                    )
        
        topic_score = calculate_topic_score(query, is_follow_up=True)
        if topic_score >= 0.1:
            return TopicRailResult(
                is_on_topic=True,
                topic_confidence=max(0.5, topic_score)
            )
    
    # Check for explicitly off-topic content
    is_off_topic, category = detect_off_topic(query)
    if is_off_topic:
        logger.info(f"Off-topic query detected: category={category}")
        return TopicRailResult(
            is_on_topic=False,
            off_topic_category=category,
            redirect_message=REDIRECT_MESSAGES.get(category, REDIRECT_MESSAGES["default"]),
            topic_confidence=0.0
        )
    
    # Calculate topic relevance score
    topic_score = calculate_topic_score(query, is_follow_up=False)
    
    # Very low score and no obvious on-topic keywords
    if topic_score < 0.2:
        # Check if it could still be a valid review question
        review_question_patterns = [
            r"(what|how|why|do|does|is|are|can|should).*\?",
            r"(tell\s+me|show\s+me|find|search|look\s+for)",
        ]
        
        is_question = any(
            re.search(p, query, re.IGNORECASE) 
            for p in review_question_patterns
        )
        
        if not is_question:
            return TopicRailResult(
                is_on_topic=False,
                redirect_message=REDIRECT_MESSAGES["default"],
                topic_confidence=topic_score
            )
    
    return TopicRailResult(
        is_on_topic=True,
        topic_confidence=max(0.5, topic_score)
    )
