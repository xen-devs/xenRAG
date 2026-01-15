"""
LLM Manager - Central interface for all LLM operations.
"""

import logging
from typing import Optional, List
from xenrag.llm.base import BaseLLM, LLMResponse
from xenrag.llm.ollama_client import OllamaClient
from xenrag.llm.gemini_client import GeminiClient
from xenrag.llm.load_balancer import LoadBalancer, LoadBalanceStrategy
from xenrag.config.settings import (
    GEMINI_MODEL, GEMINI_API_KEY, OLLAMA_URL, LLM_MODEL
)

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Central manager for LLM operations.
    Handles client initialization, load balancing, and failover.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.clients: List[BaseLLM] = []
        self.balancer: Optional[LoadBalancer] = None
        self._setup_clients()
        self._initialized = True
    
    def _setup_clients(self):
        """Initialize LLM clients based on configuration."""
        
        # Primary: Ollama (local)
        try:
            ollama = OllamaClient(
                url=OLLAMA_URL,
                model=LLM_MODEL
            )
            self.clients.append(ollama)
            logger.info(f"Registered Ollama client: {OLLAMA_URL}")
        except Exception as e:
            logger.warning(f"Failed to setup Ollama: {e}")
        
        # Secondary: Gemini (if API key available)
        gemini_key = GEMINI_API_KEY
        if gemini_key:
            try:
                gemini = GeminiClient(
                    api_key=gemini_key,
                    model=GEMINI_MODEL
                )
                self.clients.append(gemini)
                logger.info("Registered Gemini client")
            except Exception as e:
                logger.warning(f"Failed to setup Gemini: {e}")
        
        if not self.clients:
            raise RuntimeError("No LLM clients could be initialized")
        
        # Setup load balancer
        strategy = LoadBalanceStrategy.FAILOVER
        self.balancer = LoadBalancer(self.clients, strategy)
        logger.info(f"LLM Manager ready with {len(self.clients)} client(s)")
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using the best available LLM."""
        client, response = await self.balancer.execute(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )
        logger.debug(f"Generated via {client.name} in {response.latency_ms:.0f}ms")
        return response
    
    async def health_check_all(self) -> dict:
        """Check health of all clients."""
        results = {}
        for client in self.clients:
            results[client.name] = await client.health_check()
        return results
    
    def get_status(self) -> dict:
        """Get status of all clients."""
        return {
            c.name: {
                "healthy": c.is_healthy,
                "requests": c.request_count,
                "errors": c.error_count
            }
            for c in self.clients
        }


# Global instance
def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    return LLMManager()
