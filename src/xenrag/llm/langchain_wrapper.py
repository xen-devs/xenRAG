"""
LangChain-compatible wrapper for LLM Manager.
Provides ChatOllama-like interface using the multi-backend manager.
"""

from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from xenrag.llm.manager import get_llm_manager
import asyncio


class ManagedChatModel(BaseChatModel):
    """
    LangChain-compatible chat model that uses the LLM Manager.
    Supports automatic failover between Ollama and Gemini.
    """
    
    temperature: float = 0.7
    max_tokens: int = 1024
    
    @property
    def _llm_type(self) -> str:
        return "managed_chat_model"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """Synchronous generation - runs async in event loop."""
        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, stop, **kwargs)
        )
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """Async generation using LLM Manager with failover."""
        # Convert messages to prompt
        system_prompt = None
        user_prompt = ""
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                user_prompt = msg.content
            elif hasattr(msg, 'content'):
                user_prompt = msg.content
        
        # Get manager and generate
        manager = get_llm_manager()
        
        response = await manager.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Log which LLM was used
        print(f"[LLM: {response.model}] Response in {response.latency_ms:.0f}ms")
        
        # Convert to LangChain format
        ai_message = AIMessage(content=response.content)
        generation = ChatGeneration(message=ai_message)
        
        return ChatResult(generations=[generation])


def get_managed_llm(temperature: float = 0.7, max_tokens: int = 1024) -> ManagedChatModel:
    """Factory function to get a managed LLM instance."""
    return ManagedChatModel(temperature=temperature, max_tokens=max_tokens)
