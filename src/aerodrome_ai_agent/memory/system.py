"""
Memory System - Mem0-powered learning and pattern recognition

Core memory system that handles learning from experiences, pattern extraction,
intelligent recall, and memory pruning using Mem0 framework.
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime, timedelta

try:
    from mem0 import Memory
    from mem0.memory.main import MemoryBase
except ImportError:
    logging.warning("Mem0 not available, using mock implementation")
    Memory = None
    MemoryBase = None

from .config import MemoryConfig
from .patterns import PatternExtractor
from .pruning import MemoryPruner


logger = logging.getLogger(__name__)


class MemorySystem:
    """
    Intelligent memory system for learning and pattern recognition
    
    Provides capabilities for:
    - Learning from experiences and outcomes
    - Extracting patterns from repeated actions
    - Intelligent memory recall based on context
    - Automatic memory pruning and optimization
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.config.validate()
        
        # Initialize Mem0 if available
        if Memory:
            self.memory = Memory(config=config.to_mem0_config())
        else:
            self.memory = None
            logger.warning("Using mock memory implementation")
        
        # Initialize components
        self.pattern_extractor = PatternExtractor(config)
        self.memory_pruner = MemoryPruner(config)
        
        # Runtime state
        self._memory_cache: Dict[str, Any] = {}
        self._last_pruning = datetime.now()
        
        logger.info("Memory System initialized successfully")
    
    async def learn_from_experience(
        self,
        experience: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> str:
        """
        Learn from an experience and its outcome
        
        Args:
            experience: The experience data (action, context, etc.)
            outcome: The result of the experience (success, profit, etc.)
        
        Returns:
            Memory ID of stored experience
        """
        try:
            # Create structured memory entry
            memory_data = {
                "type": "experience",
                "timestamp": datetime.now().isoformat(),
                "experience": experience,
                "outcome": outcome,
                "success": outcome.get("success", False),
                "profit": outcome.get("profit", 0.0),
                "confidence": outcome.get("confidence", 0.0)
            }
            
            # Create descriptive text for embedding
            description = self._create_memory_description(memory_data)
            
            if self.memory:
                # Store in Mem0
                result = self.memory.add(
                    description,
                    user_id=self.config.user_id,
                    metadata=memory_data
                )
                memory_id = result.get("id", str(hash(description)))
            else:
                # Mock storage
                memory_id = f"memory_{hash(description)}"
                self._memory_cache[memory_id] = memory_data
            
            logger.info(f"Learned from experience: {memory_id}")
            
            # Check if we should extract patterns
            await self._check_pattern_extraction(memory_data)
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to learn from experience: {e}")
            raise
    
    async def recall_relevant_memories(
        self,
        context: Dict[str, Any],
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant memories based on context
        
        Args:
            context: Current context for memory recall
            limit: Maximum number of memories to return
        
        Returns:
            List of relevant memories with scores
        """
        limit = limit or self.config.max_query_memories
        
        try:
            # Create query from context
            query = self._create_recall_query(context)
            
            if self.memory:
                # Search using Mem0
                memories = self.memory.search(
                    query,
                    user_id=self.config.user_id,
                    limit=limit
                )
                
                # Convert to our format
                relevant_memories = []
                for memory in memories:
                    relevant_memories.append({
                        "id": memory.get("id"),
                        "content": memory.get("text", ""),
                        "metadata": memory.get("metadata", {}),
                        "score": memory.get("score", 0.0),
                        "relevance": memory.get("score", 0.0)
                    })
            else:
                # Mock search
                relevant_memories = []
                for mem_id, mem_data in self._memory_cache.items():
                    if self._is_relevant(mem_data, context):
                        relevant_memories.append({
                            "id": mem_id,
                            "content": f"Mock memory: {mem_id}",
                            "metadata": mem_data,
                            "score": 0.8,
                            "relevance": 0.8
                        })
                
                # Limit results
                relevant_memories = relevant_memories[:limit]
            
            # Filter by relevance threshold
            filtered_memories = [
                mem for mem in relevant_memories
                if mem["relevance"] >= self.config.relevance_threshold
            ]
            
            logger.info(f"Recalled {len(filtered_memories)} relevant memories")
            return filtered_memories
            
        except Exception as e:
            logger.error(f"Memory recall failed: {e}")
            return []
    
    async def extract_patterns(
        self,
        memory_type: str = None,
        min_occurrences: int = None
    ) -> List[Dict[str, Any]]:
        """
        Extract patterns from stored memories
        
        Args:
            memory_type: Type of memories to analyze
            min_occurrences: Minimum occurrences to form pattern
        
        Returns:
            List of extracted patterns
        """
        min_occurrences = min_occurrences or self.config.pattern_threshold
        
        try:
            # Get all relevant memories
            if self.memory:
                memories = self.memory.get_all(user_id=self.config.user_id)
            else:
                memories = list(self._memory_cache.values())
            
            # Filter by type if specified
            if memory_type:
                memories = [
                    m for m in memories 
                    if m.get("metadata", {}).get("type") == memory_type
                ]
            
            # Extract patterns
            patterns = await self.pattern_extractor.extract_patterns(
                memories, min_occurrences
            )
            
            logger.info(f"Extracted {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return []
    
    async def prune_memories(self) -> Dict[str, int]:
        """
        Prune old and irrelevant memories
        
        Returns:
            Statistics about pruning operation
        """
        try:
            if not self.config.pruning_enabled:
                return {"pruned": 0, "retained": 0}
            
            # Check if pruning is needed
            time_since_pruning = datetime.now() - self._last_pruning
            if time_since_pruning.total_seconds() < self.config.pruning_interval_hours * 3600:
                return {"pruned": 0, "retained": 0, "skipped": "too_recent"}
            
            # Perform pruning
            if self.memory:
                # Get all memories
                memories = self.memory.get_all(user_id=self.config.user_id)
                stats = await self.memory_pruner.prune_memories(memories, self.memory)
            else:
                # Mock pruning
                initial_count = len(self._memory_cache)
                # Remove old memories
                cutoff = datetime.now() - timedelta(days=self.config.max_memory_age_days)
                to_remove = []
                for mem_id, mem_data in self._memory_cache.items():
                    timestamp = datetime.fromisoformat(mem_data.get("timestamp", "1970-01-01"))
                    if timestamp < cutoff:
                        to_remove.append(mem_id)
                
                for mem_id in to_remove:
                    del self._memory_cache[mem_id]
                
                stats = {
                    "pruned": len(to_remove),
                    "retained": len(self._memory_cache),
                    "initial_count": initial_count
                }
            
            self._last_pruning = datetime.now()
            logger.info(f"Memory pruning completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Memory pruning failed: {e}")
            return {"error": str(e)}
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            if self.memory:
                memories = self.memory.get_all(user_id=self.config.user_id)
                total_memories = len(memories)
            else:
                total_memories = len(self._memory_cache)
            
            # Calculate tier distribution
            now = datetime.now()
            hot_cutoff = now - timedelta(days=self.config.hot_tier_days)
            warm_cutoff = now - timedelta(days=self.config.warm_tier_days)
            cold_cutoff = now - timedelta(days=self.config.cold_tier_days)
            
            tiers = {"hot": 0, "warm": 0, "cold": 0, "archive": 0}
            
            if self.memory:
                for memory in memories:
                    timestamp = datetime.fromisoformat(
                        memory.get("metadata", {}).get("timestamp", "1970-01-01")
                    )
                    if timestamp >= hot_cutoff:
                        tiers["hot"] += 1
                    elif timestamp >= warm_cutoff:
                        tiers["warm"] += 1
                    elif timestamp >= cold_cutoff:
                        tiers["cold"] += 1
                    else:
                        tiers["archive"] += 1
            else:
                for mem_data in self._memory_cache.values():
                    timestamp = datetime.fromisoformat(mem_data.get("timestamp", "1970-01-01"))
                    if timestamp >= hot_cutoff:
                        tiers["hot"] += 1
                    elif timestamp >= warm_cutoff:
                        tiers["warm"] += 1
                    elif timestamp >= cold_cutoff:
                        tiers["cold"] += 1
                    else:
                        tiers["archive"] += 1
            
            return {
                "total_memories": total_memories,
                "tier_distribution": tiers,
                "last_pruning": self._last_pruning.isoformat(),
                "config": self.config.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    def _create_memory_description(self, memory_data: Dict[str, Any]) -> str:
        """Create descriptive text for memory embedding"""
        experience = memory_data["experience"]
        outcome = memory_data["outcome"]
        
        description = f"Experience: {experience.get('action_type', 'unknown')} "
        description += f"on pool {experience.get('pool', 'unknown')} "
        description += f"with confidence {experience.get('confidence', 0)} "
        description += f"resulted in {'success' if outcome.get('success') else 'failure'} "
        description += f"with profit {outcome.get('profit', 0)}"
        
        return description
    
    def _create_recall_query(self, context: Dict[str, Any]) -> str:
        """Create search query from context"""
        query_parts = []
        
        if context.get("action_type"):
            query_parts.append(f"action type {context['action_type']}")
        
        if context.get("pool_address"):
            query_parts.append(f"pool {context['pool_address']}")
        
        if context.get("market_conditions"):
            conditions = context["market_conditions"]
            if conditions.get("volatile"):
                query_parts.append("volatile market")
            if conditions.get("high_gas"):
                query_parts.append("high gas prices")
        
        return " ".join(query_parts) or "DeFi trading experience"
    
    def _is_relevant(self, memory: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if memory is relevant to context (mock implementation)"""
        # Simple relevance check for mock implementation
        memory_type = memory.get("experience", {}).get("action_type")
        context_type = context.get("action_type")
        
        if memory_type and context_type:
            return memory_type == context_type
        
        return True  # Default to relevant
    
    async def _check_pattern_extraction(self, memory_data: Dict[str, Any]):
        """Check if pattern extraction should be triggered"""
        # This would implement logic to periodically extract patterns
        # from accumulated memories
        pass
    
    async def close(self):
        """Clean up resources"""
        if self.memory:
            # Close Mem0 connections if needed
            pass
        logger.info("Memory System closed")