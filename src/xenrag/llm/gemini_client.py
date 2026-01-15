"""
Google Gemini LLM client implementation.
Uses the new google-genai package.
"""

import time
import logging
from typing import Optional
from xenrag.llm.base import BaseLLM, LLMResponse
from xenrag.config.settings import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLM):
    """Google Gemini API client."""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        name: str = "gemini"
    ):
        super().__init__(name)
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL or "gemini-2.0-flash"
        self._client = None
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. Gemini client will not work.")
    
    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                logger.error("google-genai package not installed. Run: uv pip install google-genai")
                raise ImportError("Install with: uv pip install google-genai")
        return self._client
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using Gemini."""
        start_time = time.time()
        
        try:
            client = self._get_client()
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Use async client
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
            )
            
            latency = (time.time() - start_time) * 1000
            self.record_request()
            self.mark_healthy()
            
            return LLMResponse(
                content=response.text,
                model=self.model,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            self.mark_unhealthy()
            raise
    
    async def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        if not self.api_key:
            return False
        
        try:
            client = self._get_client()
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.model,
                    contents="Hi",
                    config={"max_output_tokens": 5}
                )
            )
            self.mark_healthy()
            return True
        except Exception as e:
            logger.debug(f"Gemini health check failed: {e}")
            self.mark_unhealthy()
            return False
