"""
JSON parsing utilities for LLM outputs.
"""

import re
import json
from typing import Optional


def parse_json_safe(text: str) -> Optional[dict]:
    """
    Safely parse JSON from LLM output.
    Handles common issues like markdown code blocks.
    """
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'(\{[\s\S]*\})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None
