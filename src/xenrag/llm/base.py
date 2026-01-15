"""
Abstract LLM interface for XenRAG.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0


class BaseLLM(ABC):
    """Abstract base class for all LLM backends."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_healthy = True
        self.request_count = 0
        self.error_count = 0
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM endpoint is healthy."""
        pass
    
    def mark_healthy(self):
        self.is_healthy = True
    
    def mark_unhealthy(self):
        self.is_healthy = False
        self.error_count += 1
    
    def record_request(self):
        self.request_count += 1
