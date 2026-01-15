"""
Load Balancer for multiple LLM endpoints.
"""

import logging
import random
from typing import List, Optional
from enum import Enum
from xenrag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    FAILOVER = "failover"


class LoadBalancer:
    """Distributes requests across multiple LLM backends."""
    
    def __init__(
        self,
        clients: List[BaseLLM],
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.FAILOVER
    ):
        self.clients = clients
        self.strategy = strategy
        self._rr_index = 0
    
    def get_healthy_clients(self) -> List[BaseLLM]:
        """Get list of healthy clients."""
        return [c for c in self.clients if c.is_healthy]
    
    def select_client(self) -> Optional[BaseLLM]:
        """Select a client based on the strategy."""
        healthy = self.get_healthy_clients()
        
        if not healthy:
            # Fallback: try all clients
            logger.warning("No healthy clients, trying all")
            healthy = self.clients
        
        if not healthy:
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            client = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return client
        
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(healthy, key=lambda c: c.request_count)
        
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(healthy)
        
        elif self.strategy == LoadBalanceStrategy.FAILOVER:
            # Return first healthy, in priority order
            return healthy[0]
        
        return healthy[0]
    
    async def execute(
        self,
        prompt: str,
        retries: int = 2,
        **kwargs
    ) -> tuple[BaseLLM, any]:
        """
        Execute request with automatic failover.
        Returns (client_used, response).
        """
        last_error = None
        tried_clients = set()
        
        for attempt in range(retries + 1):
            client = self.select_client()
            
            if client is None:
                raise RuntimeError("No LLM clients available")
            
            if client.name in tried_clients:
                # Skip already tried on retry
                untried = [c for c in self.clients if c.name not in tried_clients]
                if untried:
                    client = untried[0]
                else:
                    break
            
            tried_clients.add(client.name)
            
            try:
                logger.debug(f"Trying LLM: {client.name} (attempt {attempt + 1})")
                response = await client.generate(prompt, **kwargs)
                return client, response
                
            except Exception as e:
                last_error = e
                logger.warning(f"LLM {client.name} failed: {e}")
                client.mark_unhealthy()
        
        raise RuntimeError(f"All LLM attempts failed. Last error: {last_error}")
