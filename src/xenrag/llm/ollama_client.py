"""
Ollama LLM client implementation.
"""

import time
import logging
from typing import Optional
from langchain_ollama import ChatOllama
from xenrag.llm.base import BaseLLM, LLMResponse
from xenrag.config import settings

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLM):
    """Ollama LLM client for local inference."""
    
    def __init__(
        self,
        url: str = None,
        model: str = None,
        name: str = "ollama"
    ):
        super().__init__(name)
        self.url = url or settings.OLLAMA_URL
        self.model = model or settings.LLM_MODEL
        
    def _get_client(self, temperature: float = 0.7) -> ChatOllama:
        return ChatOllama(
            base_url=self.url,
            model=self.model,
            temperature=temperature
        )
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using Ollama."""
        start_time = time.time()
        
        try:
            client = self._get_client(temperature)
            
            messages = []
            if system_prompt:
                messages.append(("system", system_prompt))
            messages.append(("user", prompt))
            
            response = await client.ainvoke(messages)
            
            latency = (time.time() - start_time) * 1000
            self.record_request()
            self.mark_healthy()
            
            return LLMResponse(
                content=response.content,
                model=self.model,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            self.mark_unhealthy()
            raise
    
    async def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.url}/api/tags", timeout=5.0)
                healthy = response.status_code == 200
                if healthy:
                    self.mark_healthy()
                else:
                    self.mark_unhealthy()
                return healthy
        except Exception:
            self.mark_unhealthy()
            return False
