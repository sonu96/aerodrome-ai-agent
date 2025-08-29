"""
Mock implementations for Memory system components.

Provides mock objects that simulate Mem0 and memory system behavior 
without requiring actual vector databases or API keys.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock

from tests.fixtures.memory_data import (
    SUCCESSFUL_SWAP_MEMORY, FAILED_SWAP_MEMORY, LIQUIDITY_ADDITION_MEMORY,
    SUCCESSFUL_SWAP_PATTERN, HIGH_GAS_FAILURE_PATTERN, STABLE_PAIR_LP_PATTERN,
    generate_memory_sequence
)


class MockMemorySystem:
    """Mock Memory System for testing."""
    
    def __init__(self, config=None):
        self.config = config
        self._memories = {}
        self._patterns = {}
        self._should_fail = False
        self._failure_reason = "ConnectionError"
        self._query_results = {}
        
        # Initialize with some default memories
        self._init_default_memories()
    
    def _init_default_memories(self):
        """Initialize with default test memories."""
        default_memories = [
            SUCCESSFUL_SWAP_MEMORY,
            FAILED_SWAP_MEMORY,
            LIQUIDITY_ADDITION_MEMORY
        ]
        
        for memory in default_memories:
            self._memories[memory["id"]] = memory
        
        # Add some patterns
        default_patterns = [
            SUCCESSFUL_SWAP_PATTERN,
            HIGH_GAS_FAILURE_PATTERN,
            STABLE_PAIR_LP_PATTERN
        ]
        
        for pattern in default_patterns:
            self._patterns[pattern["id"]] = pattern
    
    async def learn_from_experience(
        self,
        experience: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> str:
        """Mock learning from experience."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        await asyncio.sleep(0.02)  # Simulate processing time
        
        memory_id = f"memory_{len(self._memories) + 1}"
        memory = {
            "id": memory_id,
            "type": "experience",
            "timestamp": datetime.now().isoformat(),
            "experience": experience,
            "outcome": outcome,
            "success": outcome.get("success", False),
            "profit": outcome.get("profit", 0.0),
            "confidence": outcome.get("confidence", 0.0)
        }
        
        self._memories[memory_id] = memory
        return memory_id
    
    async def recall_relevant_memories(
        self,
        context: Dict[str, Any],
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Mock memory recall."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        await asyncio.sleep(0.03)  # Simulate search time
        
        limit = limit or 10
        
        # Check for preset query results
        context_key = str(sorted(context.items()))
        if context_key in self._query_results:
            return self._query_results[context_key][:limit]
        
        # Simple relevance matching
        relevant_memories = []
        
        for memory in self._memories.values():
            if memory.get("type") != "experience":
                continue
                
            relevance = self._calculate_relevance(memory, context)
            if relevance >= 0.6:  # Default relevance threshold
                relevant_memories.append({
                    "id": memory["id"],
                    "content": self._create_memory_description(memory),
                    "metadata": memory,
                    "score": relevance,
                    "relevance": relevance
                })
        
        # Sort by relevance and limit
        relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant_memories[:limit]
    
    async def extract_patterns(
        self,
        memory_type: str = None,
        min_occurrences: int = None
    ) -> List[Dict[str, Any]]:
        """Mock pattern extraction."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        await asyncio.sleep(0.05)  # Simulate analysis time
        
        patterns = list(self._patterns.values())
        
        # Filter by type if specified
        if memory_type:
            patterns = [p for p in patterns if memory_type in p.get("pattern_type", "")]
        
        return patterns
    
    async def prune_memories(self) -> Dict[str, int]:
        """Mock memory pruning."""
        if self._should_fail:
            return {"error": self._failure_reason}
        
        await asyncio.sleep(0.04)  # Simulate pruning time
        
        initial_count = len(self._memories)
        
        # Simulate pruning old memories
        cutoff = datetime.now() - timedelta(days=30)
        pruned_count = 0
        
        to_remove = []
        for mem_id, memory in self._memories.items():
            try:
                timestamp = datetime.fromisoformat(memory.get("timestamp", "1970-01-01"))
                if timestamp < cutoff:
                    to_remove.append(mem_id)
                    pruned_count += 1
            except ValueError:
                continue
        
        for mem_id in to_remove:
            del self._memories[mem_id]
        
        return {
            "pruned": pruned_count,
            "retained": len(self._memories),
            "initial_count": initial_count
        }
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Mock memory statistics."""
        if self._should_fail:
            return {"error": self._failure_reason}
        
        await asyncio.sleep(0.01)
        
        # Calculate tier distribution
        now = datetime.now()
        hot_cutoff = now - timedelta(days=7)
        warm_cutoff = now - timedelta(days=30)
        cold_cutoff = now - timedelta(days=60)
        
        tiers = {"hot": 0, "warm": 0, "cold": 0, "archive": 0}
        
        for memory in self._memories.values():
            try:
                timestamp = datetime.fromisoformat(memory.get("timestamp", "1970-01-01"))
                if timestamp >= hot_cutoff:
                    tiers["hot"] += 1
                elif timestamp >= warm_cutoff:
                    tiers["warm"] += 1
                elif timestamp >= cold_cutoff:
                    tiers["cold"] += 1
                else:
                    tiers["archive"] += 1
            except ValueError:
                tiers["archive"] += 1
        
        return {
            "total_memories": len(self._memories),
            "tier_distribution": tiers,
            "last_pruning": datetime.now().isoformat(),
            "patterns_count": len(self._patterns),
            "config": self.config.to_dict() if self.config else {}
        }
    
    def _calculate_relevance(self, memory: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate relevance between memory and context."""
        relevance = 0.0
        
        experience = memory.get("experience", {})
        
        # Action type matching
        if context.get("action_type") == experience.get("action_type"):
            relevance += 0.4
        
        # Pool address matching
        if context.get("pool_address") == experience.get("pool"):
            relevance += 0.3
        
        # Market conditions matching
        context_conditions = context.get("market_conditions", {})
        memory_conditions = experience.get("market_conditions", {})
        
        if context_conditions and memory_conditions:
            if context_conditions.get("volatile") == memory_conditions.get("volatile"):
                relevance += 0.2
            if abs(context_conditions.get("volatility", 0) - memory_conditions.get("volatility", 0)) < 0.1:
                relevance += 0.1
        
        return min(relevance, 1.0)
    
    def _create_memory_description(self, memory: Dict[str, Any]) -> str:
        """Create description for memory."""
        experience = memory["experience"]
        outcome = memory["outcome"]
        
        description = f"Experience: {experience.get('action_type', 'unknown')} "
        description += f"on pool {experience.get('pool', 'unknown')} "
        description += f"with confidence {experience.get('confidence', 0):.2f} "
        description += f"resulted in {'success' if outcome.get('success') else 'failure'} "
        description += f"with profit {outcome.get('profit', 0):.2f}"
        
        return description
    
    # Test utilities
    def add_memory(self, memory: Dict[str, Any]):
        """Add a memory for testing."""
        self._memories[memory["id"]] = memory
    
    def add_pattern(self, pattern: Dict[str, Any]):
        """Add a pattern for testing."""
        self._patterns[pattern["id"]] = pattern
    
    def set_should_fail(self, should_fail: bool, reason: str = "ConnectionError"):
        """Configure mock to simulate failures."""
        self._should_fail = should_fail
        self._failure_reason = reason
    
    def set_query_result(self, context: Dict[str, Any], result: List[Dict[str, Any]]):
        """Set custom query result for testing."""
        context_key = str(sorted(context.items()))
        self._query_results[context_key] = result
    
    def clear_memories(self):
        """Clear all memories for testing."""
        self._memories = {}
    
    def clear_patterns(self):
        """Clear all patterns for testing."""
        self._patterns = {}
    
    def get_memory_count(self) -> int:
        """Get total memory count."""
        return len(self._memories)
    
    def get_pattern_count(self) -> int:
        """Get total pattern count."""
        return len(self._patterns)
    
    async def close(self):
        """Mock cleanup."""
        pass


class MockPatternExtractor:
    """Mock pattern extractor for testing."""
    
    def __init__(self, config=None):
        self.config = config
        self._should_fail = False
        self._failure_reason = "AnalysisError"
        self._custom_patterns = []
    
    async def extract_patterns(
        self, 
        memories: List[Dict[str, Any]], 
        min_occurrences: int = 3
    ) -> List[Dict[str, Any]]:
        """Mock pattern extraction."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        await asyncio.sleep(0.1)  # Simulate analysis time
        
        if self._custom_patterns:
            return self._custom_patterns
        
        # Simple pattern extraction simulation
        patterns = []
        
        # Group memories by action type
        action_groups = {}
        for memory in memories:
            if memory.get("type") != "experience":
                continue
            
            action_type = memory.get("experience", {}).get("action_type", "UNKNOWN")
            if action_type not in action_groups:
                action_groups[action_type] = []
            action_groups[action_type].append(memory)
        
        # Create patterns for groups with enough occurrences
        for action_type, group_memories in action_groups.items():
            if len(group_memories) >= min_occurrences:
                success_rate = sum(1 for m in group_memories 
                                 if m.get("outcome", {}).get("success", False)) / len(group_memories)
                avg_profit = sum(m.get("outcome", {}).get("profit", 0) 
                               for m in group_memories) / len(group_memories)
                
                pattern = {
                    "id": f"pattern_{action_type.lower()}",
                    "type": "behavioral_pattern",
                    "pattern_type": f"{action_type.lower()}_pattern",
                    "occurrences": len(group_memories),
                    "confidence": min(0.95, success_rate + 0.1),
                    "conditions": {
                        "action_type": action_type,
                        "min_confidence": 0.5
                    },
                    "outcomes": {
                        "success_rate": success_rate,
                        "avg_profit": avg_profit
                    },
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
                patterns.append(pattern)
        
        return patterns
    
    def set_should_fail(self, should_fail: bool, reason: str = "AnalysisError"):
        """Configure mock to simulate failures."""
        self._should_fail = should_fail
        self._failure_reason = reason
    
    def set_custom_patterns(self, patterns: List[Dict[str, Any]]):
        """Set custom patterns for testing."""
        self._custom_patterns = patterns


class MockMemoryPruner:
    """Mock memory pruner for testing."""
    
    def __init__(self, config=None):
        self.config = config
        self._should_fail = False
        self._failure_reason = "PruningError"
        self._pruning_stats = None
    
    async def prune_memories(
        self, 
        memories: List[Dict[str, Any]], 
        memory_system
    ) -> Dict[str, int]:
        """Mock memory pruning."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        await asyncio.sleep(0.05)  # Simulate pruning time
        
        if self._pruning_stats:
            return self._pruning_stats
        
        # Simple pruning simulation
        initial_count = len(memories)
        cutoff = datetime.now() - timedelta(days=30)
        
        pruned_count = 0
        for memory in memories:
            try:
                timestamp = datetime.fromisoformat(memory.get("timestamp", "1970-01-01"))
                if timestamp < cutoff:
                    pruned_count += 1
            except ValueError:
                pruned_count += 1  # Count invalid timestamps as pruned
        
        return {
            "pruned": pruned_count,
            "retained": initial_count - pruned_count,
            "initial_count": initial_count
        }
    
    def set_should_fail(self, should_fail: bool, reason: str = "PruningError"):
        """Configure mock to simulate failures."""
        self._should_fail = should_fail
        self._failure_reason = reason
    
    def set_pruning_stats(self, stats: Dict[str, int]):
        """Set custom pruning stats for testing."""
        self._pruning_stats = stats


class MockMem0Memory:
    """Mock Mem0 Memory class for testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._memories = {}
        self._should_fail = False
        self._failure_reason = "APIError"
    
    def add(self, text: str, user_id: str = None, metadata: Dict[str, Any] = None):
        """Mock add memory."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        memory_id = f"mem_{len(self._memories) + 1}"
        memory = {
            "id": memory_id,
            "text": text,
            "user_id": user_id,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self._memories[memory_id] = memory
        return {"id": memory_id}
    
    def search(self, query: str, user_id: str = None, limit: int = 10):
        """Mock search memories."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        # Simple text matching
        results = []
        query_lower = query.lower()
        
        for memory in self._memories.values():
            if user_id and memory.get("user_id") != user_id:
                continue
            
            text_lower = memory.get("text", "").lower()
            if query_lower in text_lower:
                results.append({
                    "id": memory["id"],
                    "text": memory["text"],
                    "metadata": memory.get("metadata", {}),
                    "score": 0.8  # Mock relevance score
                })
        
        return results[:limit]
    
    def get_all(self, user_id: str = None):
        """Mock get all memories."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        memories = list(self._memories.values())
        
        if user_id:
            memories = [m for m in memories if m.get("user_id") == user_id]
        
        return memories
    
    def delete(self, memory_id: str):
        """Mock delete memory."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False
    
    def set_should_fail(self, should_fail: bool, reason: str = "APIError"):
        """Configure mock to simulate failures."""
        self._should_fail = should_fail
        self._failure_reason = reason
    
    def clear(self):
        """Clear all memories."""
        self._memories = {}
    
    def get_memory_count(self) -> int:
        """Get memory count."""
        return len(self._memories)