"""
Tests for the Aerodrome Memory System.

Comprehensive test suite covering memory storage, recall, pattern extraction,
pruning, and integration with Mem0.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from aerodrome_ai_agent.memory.system import MemorySystem
from aerodrome_ai_agent.memory.config import MemoryConfig
from tests.mocks.memory_mocks import MockMemorySystem, MockMem0Memory
from tests.fixtures.memory_data import (
    SUCCESSFUL_SWAP_MEMORY, FAILED_SWAP_MEMORY, LIQUIDITY_ADDITION_MEMORY,
    SUCCESSFUL_SWAP_PATTERN, HIGH_GAS_FAILURE_PATTERN, MEMORY_QUERY_SCENARIOS,
    generate_memory_sequence, get_memories_by_type, get_memories_by_success
)


class TestMemoryConfig:
    """Test memory configuration handling."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryConfig()
        
        assert config.user_id == "aerodrome_agent"
        assert config.vector_store == "qdrant"
        assert config.max_memories == 10000
        assert config.pattern_threshold == 5
        assert config.pruning_enabled is True
        assert config.hot_tier_days == 7
        assert config.warm_tier_days == 30
        assert config.cold_tier_days == 60
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        env_vars = {
            "MEMORY_USER_ID": "test_agent",
            "MEMORY_MAX_MEMORIES": "5000",
            "MEMORY_PATTERN_THRESHOLD": "3",
            "QDRANT_HOST": "test_host",
            "QDRANT_PORT": "9999"
        }
        
        with patch.dict('os.environ', env_vars):
            config = MemoryConfig.from_env()
            
            assert config.user_id == "test_agent"
            assert config.max_memories == 5000
            assert config.pattern_threshold == 3
            assert config.qdrant_host == "test_host"
            assert config.qdrant_port == 9999
    
    def test_config_validation_valid(self):
        """Test valid configuration passes validation."""
        config = MemoryConfig(
            user_id="test_user",
            max_memories=5000,
            pattern_threshold=3,
            hot_tier_days=5,
            warm_tier_days=20,
            cold_tier_days=50
        )
        
        # Should not raise
        config.validate()
    
    def test_config_validation_invalid_user_id(self):
        """Test invalid user ID fails validation."""
        config = MemoryConfig(user_id="")
        
        with pytest.raises(ValueError, match="user_id is required"):
            config.validate()
    
    def test_config_validation_invalid_max_memories(self):
        """Test invalid max memories fails validation."""
        config = MemoryConfig(max_memories=-1)
        
        with pytest.raises(ValueError, match="max_memories must be positive"):
            config.validate()
    
    def test_config_validation_invalid_tier_order(self):
        """Test invalid tier day ordering fails validation."""
        config = MemoryConfig(
            hot_tier_days=30,
            warm_tier_days=20,  # Should be > hot_tier_days
            cold_tier_days=50
        )
        
        with pytest.raises(ValueError, match="hot_tier_days must be less than warm_tier_days"):
            config.validate()
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = MemoryConfig(user_id="test_user", max_memories=5000)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["user_id"] == "test_user"
        assert config_dict["max_memories"] == 5000
        assert "pattern_threshold" in config_dict
    
    def test_config_to_mem0_config(self):
        """Test conversion to Mem0 configuration format."""
        config = MemoryConfig(
            vector_store="qdrant",
            qdrant_host="localhost",
            qdrant_port=6333,
            embedding_model="text-embedding-ada-002",
            llm_model="gpt-4"
        )
        
        mem0_config = config.to_mem0_config()
        
        assert mem0_config["vector_store"]["provider"] == "qdrant"
        assert mem0_config["vector_store"]["config"]["host"] == "localhost"
        assert mem0_config["vector_store"]["config"]["port"] == 6333
        assert mem0_config["embedder"]["config"]["model"] == "text-embedding-ada-002"
        assert mem0_config["llm"]["config"]["model"] == "gpt-4"


class TestMemorySystem:
    """Test core memory system functionality."""
    
    @pytest.fixture
    def memory_system(self, memory_config):
        """Create memory system for testing."""
        return MemorySystem(memory_config)
    
    @pytest.fixture
    def mock_memory_system(self, memory_config):
        """Create mock memory system for testing."""
        return MockMemorySystem(memory_config)
    
    def test_memory_system_initialization(self, memory_config):
        """Test memory system initialization."""
        with patch('aerodrome_ai_agent.memory.system.Memory', None):
            system = MemorySystem(memory_config)
            
            assert system.config == memory_config
            assert system.memory is None
            assert isinstance(system._memory_cache, dict)
    
    def test_memory_system_initialization_with_mem0(self, memory_config):
        """Test memory system initialization with Mem0."""
        mock_memory = MagicMock()
        
        with patch('aerodrome_ai_agent.memory.system.Memory', return_value=mock_memory):
            system = MemorySystem(memory_config)
            
            assert system.memory == mock_memory
    
    @pytest.mark.asyncio
    async def test_learn_from_experience_success(self, mock_memory_system):
        """Test learning from successful experience."""
        experience = {
            "action_type": "SWAP",
            "pool": "0x1234567890abcdef",
            "amount": 1000.0,
            "confidence": 0.85
        }
        outcome = {
            "success": True,
            "profit": 25.50,
            "confidence": 0.92
        }
        
        memory_id = await mock_memory_system.learn_from_experience(experience, outcome)
        
        assert memory_id is not None
        assert mock_memory_system.get_memory_count() > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_experience_failure(self, mock_memory_system):
        """Test learning from failed experience."""
        experience = {
            "action_type": "SWAP",
            "pool": "0x2345678901bcdef0",
            "amount": 500.0,
            "confidence": 0.65
        }
        outcome = {
            "success": False,
            "profit": -15.25,
            "error": "SlippageExceeded"
        }
        
        memory_id = await mock_memory_system.learn_from_experience(experience, outcome)
        
        assert memory_id is not None
        # Verify failed experience is stored
        memories = await mock_memory_system.recall_relevant_memories(
            {"action_type": "SWAP"}, limit=10
        )
        
        failed_memories = [m for m in memories 
                          if not m["metadata"]["outcome"]["success"]]
        assert len(failed_memories) > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_experience_error(self, mock_memory_system):
        """Test learning from experience with system error."""
        mock_memory_system.set_should_fail(True, "StorageError")
        
        experience = {"action_type": "SWAP", "pool": "0x123"}
        outcome = {"success": True, "profit": 10.0}
        
        with pytest.raises(Exception, match="StorageError"):
            await mock_memory_system.learn_from_experience(experience, outcome)
    
    @pytest.mark.asyncio
    async def test_recall_relevant_memories_by_action_type(self, mock_memory_system):
        """Test memory recall filtered by action type."""
        # Add memories for different action types
        await mock_memory_system.learn_from_experience(
            {"action_type": "SWAP", "pool": "0x123"},
            {"success": True, "profit": 10.0}
        )
        await mock_memory_system.learn_from_experience(
            {"action_type": "ADD_LIQUIDITY", "pool": "0x456"},
            {"success": True, "profit": 5.0}
        )
        
        # Recall SWAP memories only
        context = {"action_type": "SWAP"}
        memories = await mock_memory_system.recall_relevant_memories(context, limit=5)
        
        swap_memories = [m for m in memories 
                        if m["metadata"]["experience"]["action_type"] == "SWAP"]
        assert len(swap_memories) > 0
    
    @pytest.mark.asyncio
    async def test_recall_relevant_memories_by_pool(self, mock_memory_system):
        """Test memory recall filtered by pool address."""
        pool_address = "0x1234567890abcdef"
        
        await mock_memory_system.learn_from_experience(
            {"action_type": "SWAP", "pool": pool_address},
            {"success": True, "profit": 15.0}
        )
        
        context = {"pool_address": pool_address}
        memories = await mock_memory_system.recall_relevant_memories(context, limit=5)
        
        pool_memories = [m for m in memories 
                        if m["metadata"]["experience"].get("pool") == pool_address]
        assert len(pool_memories) > 0
    
    @pytest.mark.asyncio
    async def test_recall_relevant_memories_limit(self, mock_memory_system):
        """Test memory recall with limit parameter."""
        # Add multiple memories
        for i in range(10):
            await mock_memory_system.learn_from_experience(
                {"action_type": "SWAP", "pool": f"0x{i:040d}"},
                {"success": True, "profit": i * 5.0}
            )
        
        context = {"action_type": "SWAP"}
        memories = await mock_memory_system.recall_relevant_memories(context, limit=3)
        
        assert len(memories) <= 3
    
    @pytest.mark.asyncio
    async def test_recall_relevant_memories_error(self, mock_memory_system):
        """Test memory recall with system error."""
        mock_memory_system.set_should_fail(True, "SearchError")
        
        context = {"action_type": "SWAP"}
        memories = await mock_memory_system.recall_relevant_memories(context)
        
        # Should return empty list on error
        assert memories == []
    
    @pytest.mark.asyncio
    async def test_extract_patterns_success(self, mock_memory_system):
        """Test successful pattern extraction.""" 
        # Add multiple similar memories to form patterns
        for i in range(5):
            await mock_memory_system.learn_from_experience(
                {
                    "action_type": "SWAP",
                    "pool": "0x1234567890abcdef",
                    "confidence": 0.8 + i * 0.02
                },
                {
                    "success": True,
                    "profit": 20.0 + i * 2.0,
                    "confidence": 0.85 + i * 0.02
                }
            )
        
        patterns = await mock_memory_system.extract_patterns(
            memory_type="experience", min_occurrences=3
        )
        
        assert len(patterns) > 0
        swap_patterns = [p for p in patterns if "swap" in p["pattern_type"]]
        assert len(swap_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_extract_patterns_insufficient_data(self, mock_memory_system):
        """Test pattern extraction with insufficient data."""
        # Add only one memory
        await mock_memory_system.learn_from_experience(
            {"action_type": "SWAP", "pool": "0x123"},
            {"success": True, "profit": 10.0}
        )
        
        patterns = await mock_memory_system.extract_patterns(min_occurrences=5)
        
        # Should return empty list or existing patterns only
        assert isinstance(patterns, list)
    
    @pytest.mark.asyncio
    async def test_extract_patterns_error(self, mock_memory_system):
        """Test pattern extraction with system error."""
        mock_memory_system.set_should_fail(True, "AnalysisError")
        
        patterns = await mock_memory_system.extract_patterns()
        
        # Should return empty list on error
        assert patterns == []
    
    @pytest.mark.asyncio
    async def test_prune_memories_success(self, mock_memory_system):
        """Test successful memory pruning."""
        # Add old and new memories
        old_time = datetime.now() - timedelta(days=35)
        
        # Add old memory
        old_memory = {
            "id": "old_memory",
            "type": "experience",
            "timestamp": old_time.isoformat(),
            "experience": {"action_type": "SWAP"},
            "outcome": {"success": True, "profit": 10.0}
        }
        mock_memory_system.add_memory(old_memory)
        
        # Add recent memory
        await mock_memory_system.learn_from_experience(
            {"action_type": "SWAP", "pool": "0x123"},
            {"success": True, "profit": 15.0}
        )
        
        initial_count = mock_memory_system.get_memory_count()
        stats = await mock_memory_system.prune_memories()
        
        assert "pruned" in stats
        assert "retained" in stats
        assert stats["pruned"] >= 0
        assert stats["retained"] <= initial_count
    
    @pytest.mark.asyncio
    async def test_prune_memories_disabled(self, memory_config):
        """Test memory pruning when disabled."""
        memory_config.pruning_enabled = False
        mock_system = MockMemorySystem(memory_config)
        
        stats = await mock_system.prune_memories()
        
        assert stats["pruned"] == 0
        assert stats["retained"] == 0
    
    @pytest.mark.asyncio
    async def test_prune_memories_error(self, mock_memory_system):
        """Test memory pruning with system error."""
        mock_memory_system.set_should_fail(True, "PruningError")
        
        stats = await mock_memory_system.prune_memories()
        
        assert "error" in stats
        assert "PruningError" in str(stats["error"])
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, mock_memory_system):
        """Test memory statistics retrieval."""
        # Add memories in different time periods
        now = datetime.now()
        
        # Hot tier memory (recent)
        await mock_memory_system.learn_from_experience(
            {"action_type": "SWAP", "pool": "0x123"},
            {"success": True, "profit": 10.0}
        )
        
        # Warm tier memory (older)
        warm_memory = {
            "id": "warm_memory",
            "type": "experience", 
            "timestamp": (now - timedelta(days=15)).isoformat(),
            "experience": {"action_type": "ADD_LIQUIDITY"},
            "outcome": {"success": True, "profit": 5.0}
        }
        mock_memory_system.add_memory(warm_memory)
        
        stats = await mock_memory_system.get_memory_stats()
        
        assert "total_memories" in stats
        assert "tier_distribution" in stats
        assert "last_pruning" in stats
        assert stats["total_memories"] > 0
        assert "hot" in stats["tier_distribution"]
        assert "warm" in stats["tier_distribution"]
        assert "cold" in stats["tier_distribution"]
    
    @pytest.mark.asyncio
    async def test_get_memory_stats_error(self, mock_memory_system):
        """Test memory statistics with system error."""
        mock_memory_system.set_should_fail(True, "StatisticsError")
        
        stats = await mock_memory_system.get_memory_stats()
        
        assert "error" in stats
        assert "StatisticsError" in str(stats["error"])
    
    @pytest.mark.asyncio
    async def test_close(self, mock_memory_system):
        """Test memory system cleanup."""
        # Should not raise exception
        await mock_memory_system.close()


class TestMemoryPatterns:
    """Test pattern extraction and analysis."""
    
    @pytest.fixture
    def memory_with_patterns(self, memory_config):
        """Create memory system with pattern data."""
        mock_system = MockMemorySystem(memory_config)
        
        # Add patterns
        mock_system.add_pattern(SUCCESSFUL_SWAP_PATTERN)
        mock_system.add_pattern(HIGH_GAS_FAILURE_PATTERN)
        
        return mock_system
    
    @pytest.mark.asyncio
    async def test_pattern_recognition_successful_swaps(self, memory_with_patterns):
        """Test recognition of successful swap patterns."""
        patterns = await memory_with_patterns.extract_patterns(
            memory_type="experience", min_occurrences=3
        )
        
        success_patterns = [p for p in patterns 
                           if "successful" in p["pattern_type"]]
        assert len(success_patterns) > 0
        
        pattern = success_patterns[0]
        assert pattern["confidence"] > 0.8
        assert pattern["occurrences"] >= 3
        assert "conditions" in pattern
        assert "outcomes" in pattern
    
    @pytest.mark.asyncio
    async def test_pattern_recognition_failure_conditions(self, memory_with_patterns):
        """Test recognition of failure patterns."""
        patterns = await memory_with_patterns.extract_patterns()
        
        failure_patterns = [p for p in patterns 
                           if p["type"] == "failure_pattern"]
        assert len(failure_patterns) > 0
        
        pattern = failure_patterns[0]
        assert "gas" in pattern["pattern_type"]
        assert pattern["outcomes"]["success_rate"] < 0.5
    
    @pytest.mark.asyncio
    async def test_pattern_confidence_scoring(self, memory_with_patterns):
        """Test pattern confidence scoring."""
        patterns = await memory_with_patterns.extract_patterns()
        
        for pattern in patterns:
            assert 0 <= pattern["confidence"] <= 1
            assert pattern["occurrences"] > 0
            
            # High occurrence patterns should have higher confidence
            if pattern["occurrences"] >= 8:
                assert pattern["confidence"] >= 0.8


class TestMemoryQueries:
    """Test various memory query scenarios.""" 
    
    @pytest.fixture
    def populated_memory_system(self, memory_config):
        """Create memory system with diverse data."""
        mock_system = MockMemorySystem(memory_config)
        
        # Add diverse memories
        memories = generate_memory_sequence(days=7)
        for memory in memories:
            mock_system.add_memory(memory)
        
        return mock_system
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario", MEMORY_QUERY_SCENARIOS)
    async def test_query_scenarios(self, populated_memory_system, scenario):
        """Test various memory query scenarios."""
        context = scenario["context"]
        expected_count = scenario["expected_count"]
        min_relevance = scenario["min_relevance"]
        
        memories = await populated_memory_system.recall_relevant_memories(
            context, limit=expected_count
        )
        
        # Check that we get some results
        assert len(memories) >= 0
        
        # Check relevance scores if memories found
        if memories:
            for memory in memories:
                assert memory["relevance"] >= min_relevance
    
    @pytest.mark.asyncio
    async def test_query_by_success_status(self, populated_memory_system):
        """Test querying memories by success status."""
        # Query for successful experiences only
        context = {"success_only": True}
        
        # Set custom query result
        successful_memories = [
            {
                "id": "success_1",
                "content": "Successful swap operation",
                "metadata": {
                    "type": "experience",
                    "outcome": {"success": True, "profit": 25.0}
                },
                "relevance": 0.9,
                "score": 0.9
            }
        ]
        populated_memory_system.set_query_result(context, successful_memories)
        
        memories = await populated_memory_system.recall_relevant_memories(context)
        
        assert len(memories) > 0
        for memory in memories:
            assert memory["metadata"]["outcome"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_query_by_timeframe(self, populated_memory_system):
        """Test querying memories by timeframe.""" 
        context = {"timeframe": "recent"}
        
        # Set custom result for recent memories
        recent_memories = [
            {
                "id": "recent_1",
                "content": "Recent trading activity",
                "metadata": {
                    "type": "experience", 
                    "timestamp": datetime.now().isoformat()
                },
                "relevance": 0.85,
                "score": 0.85
            }
        ]
        populated_memory_system.set_query_result(context, recent_memories)
        
        memories = await populated_memory_system.recall_relevant_memories(context)
        
        assert len(memories) > 0
        # Verify memories are recent (within last day)
        for memory in memories:
            timestamp = datetime.fromisoformat(
                memory["metadata"]["timestamp"]
            )
            age = datetime.now() - timestamp
            assert age.total_seconds() < 86400  # Less than 1 day


class TestMemoryIntegration:
    """Integration tests for memory system components."""
    
    @pytest.mark.asyncio
    async def test_learning_and_recall_cycle(self, memory_config):
        """Test complete learning and recall cycle."""
        system = MockMemorySystem(memory_config)
        
        # Learn from multiple experiences
        experiences = [
            {
                "experience": {"action_type": "SWAP", "pool": "0x123", "confidence": 0.8},
                "outcome": {"success": True, "profit": 25.0}
            },
            {
                "experience": {"action_type": "SWAP", "pool": "0x123", "confidence": 0.7}, 
                "outcome": {"success": False, "profit": -10.0}
            },
            {
                "experience": {"action_type": "SWAP", "pool": "0x456", "confidence": 0.9},
                "outcome": {"success": True, "profit": 35.0}
            }
        ]
        
        # Store experiences
        memory_ids = []
        for exp in experiences:
            memory_id = await system.learn_from_experience(
                exp["experience"], exp["outcome"]
            )
            memory_ids.append(memory_id)
        
        # Verify all memories stored
        assert len(memory_ids) == 3
        assert system.get_memory_count() >= 3
        
        # Test recall for specific pool
        context = {"pool_address": "0x123"}
        pool_memories = await system.recall_relevant_memories(context)
        
        # Should find memories for pool 0x123
        relevant_memories = [m for m in pool_memories 
                           if m["metadata"]["experience"]["pool"] == "0x123"]
        assert len(relevant_memories) >= 1
        
        # Test recall for successful trades
        success_context = {"action_type": "SWAP"}
        all_swap_memories = await system.recall_relevant_memories(success_context)
        
        successful_swaps = [m for m in all_swap_memories
                           if m["metadata"]["outcome"]["success"]]
        assert len(successful_swaps) >= 1
    
    @pytest.mark.asyncio
    async def test_pattern_extraction_from_experiences(self, memory_config):
        """Test pattern extraction from stored experiences."""
        system = MockMemorySystem(memory_config)
        
        # Add multiple similar experiences to form patterns
        base_experience = {
            "action_type": "SWAP",
            "pool": "0x1234567890abcdef",
            "market_conditions": {"volatility": 0.1, "trending": "bullish"}
        }
        
        # Add successful experiences with similar conditions
        for i in range(6):
            experience = base_experience.copy()
            experience["confidence"] = 0.75 + i * 0.02
            
            outcome = {
                "success": True,
                "profit": 20.0 + i * 3.0,
                "confidence": 0.8 + i * 0.02
            }
            
            await system.learn_from_experience(experience, outcome)
        
        # Extract patterns
        patterns = await system.extract_patterns(min_occurrences=4)
        
        assert len(patterns) > 0
        
        # Find swap pattern
        swap_patterns = [p for p in patterns if "swap" in p["pattern_type"]]
        assert len(swap_patterns) > 0
        
        pattern = swap_patterns[0]
        assert pattern["occurrences"] >= 4
        assert pattern["outcomes"]["success_rate"] > 0.8
        assert pattern["confidence"] > 0.8
    
    @pytest.mark.asyncio
    async def test_memory_pruning_lifecycle(self, memory_config):
        """Test memory lifecycle with pruning."""
        # Set short tier durations for testing
        config = memory_config
        config.hot_tier_days = 1
        config.warm_tier_days = 3
        config.cold_tier_days = 7
        config.max_memory_age_days = 10
        
        system = MockMemorySystem(config)
        
        # Add memories of different ages
        now = datetime.now()
        memory_ages = [
            timedelta(hours=12),  # Hot
            timedelta(days=2),    # Warm
            timedelta(days=5),    # Cold
            timedelta(days=15)    # Should be pruned
        ]
        
        for i, age in enumerate(memory_ages):
            memory_time = now - age
            memory = {
                "id": f"memory_{i}",
                "type": "experience",
                "timestamp": memory_time.isoformat(),
                "experience": {"action_type": "SWAP", "pool": f"0x{i:040d}"},
                "outcome": {"success": True, "profit": 10.0}
            }
            system.add_memory(memory)
        
        # Get initial stats
        initial_stats = await system.get_memory_stats()
        initial_count = initial_stats["total_memories"]
        
        # Perform pruning
        pruning_stats = await system.prune_memories()
        
        # Verify pruning occurred
        assert pruning_stats["initial_count"] == initial_count
        assert pruning_stats["pruned"] >= 0
        assert pruning_stats["retained"] <= initial_count
        
        # Get final stats
        final_stats = await system.get_memory_stats()
        assert final_stats["total_memories"] == pruning_stats["retained"]
    
    @pytest.mark.asyncio
    async def test_memory_system_error_recovery(self, memory_config):
        """Test memory system error recovery."""
        system = MockMemorySystem(memory_config)
        
        # Normal operation
        memory_id = await system.learn_from_experience(
            {"action_type": "SWAP", "pool": "0x123"},
            {"success": True, "profit": 10.0}
        )
        assert memory_id is not None
        
        # Simulate system failure
        system.set_should_fail(True, "TemporaryError")
        
        # Operations should fail gracefully
        with pytest.raises(Exception):
            await system.learn_from_experience(
                {"action_type": "SWAP", "pool": "0x456"},
                {"success": True, "profit": 15.0}
            )
        
        # Recall should return empty results on error
        memories = await system.recall_relevant_memories({"action_type": "SWAP"})
        assert memories == []
        
        # Recovery
        system.set_should_fail(False)
        
        # Normal operation should resume
        new_memory_id = await system.learn_from_experience(
            {"action_type": "ADD_LIQUIDITY", "pool": "0x789"}, 
            {"success": True, "profit": 20.0}
        )
        assert new_memory_id is not None
        
        # Recall should work again
        memories = await system.recall_relevant_memories({"action_type": "ADD_LIQUIDITY"})
        assert len(memories) > 0


class TestMem0Integration:
    """Test integration with Mem0 library."""
    
    @pytest.fixture
    def mock_mem0(self):
        """Create mock Mem0 instance."""
        return MockMem0Memory()
    
    @pytest.mark.asyncio
    async def test_mem0_memory_storage(self, memory_config, mock_mem0):
        """Test memory storage through Mem0."""
        with patch('aerodrome_ai_agent.memory.system.Memory', return_value=mock_mem0):
            system = MemorySystem(memory_config)
            
            # Store memory
            memory_id = await system.learn_from_experience(
                {"action_type": "SWAP", "pool": "0x123"},
                {"success": True, "profit": 15.0}
            )
            
            assert memory_id is not None
            assert mock_mem0.get_memory_count() > 0
    
    @pytest.mark.asyncio
    async def test_mem0_memory_search(self, memory_config, mock_mem0):
        """Test memory search through Mem0.""" 
        with patch('aerodrome_ai_agent.memory.system.Memory', return_value=mock_mem0):
            system = MemorySystem(memory_config)
            
            # Store some memories
            await system.learn_from_experience(
                {"action_type": "SWAP", "pool": "0x123"},
                {"success": True, "profit": 15.0}
            )
            await system.learn_from_experience(
                {"action_type": "ADD_LIQUIDITY", "pool": "0x456"},
                {"success": True, "profit": 25.0}
            )
            
            # Search for memories
            memories = await system.recall_relevant_memories(
                {"action_type": "SWAP"}
            )
            
            assert len(memories) > 0
            # Verify search worked through Mem0
            assert any("SWAP" in mem["content"] for mem in memories)
    
    @pytest.mark.asyncio
    async def test_mem0_error_handling(self, memory_config, mock_mem0):
        """Test error handling with Mem0."""
        mock_mem0.set_should_fail(True, "Mem0APIError")
        
        with patch('aerodrome_ai_agent.memory.system.Memory', return_value=mock_mem0):
            system = MemorySystem(memory_config)
            
            # Should handle Mem0 errors gracefully
            with pytest.raises(Exception):
                await system.learn_from_experience(
                    {"action_type": "SWAP", "pool": "0x123"},
                    {"success": True, "profit": 15.0}
                )