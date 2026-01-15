"""
Text Normalization Module
Cleans and standardizes raw text for ingestion.
"""

import re
import html
from typing import Optional


def normalize_text(text: str) -> str:
    """
    Normalize text by cleaning HTML, Unicode, and whitespace.
    """
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Normalize Unicode quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', '...')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def clean_for_embedding(text: str) -> str:
    """
    Further clean text for embedding generation.
    Removes special characters that don't add semantic value.
    """
    text = normalize_text(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!?]{2,}', '!', text)
    text = re.sub(r'\.{3,}', '...', text)
    
    return text.strip()
