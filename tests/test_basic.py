"""
Basic tests to verify project structure and imports
"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_imports():
    """Test that main components can be imported"""
    
    # Test main package import
    import aerodrome_ai_agent
    assert aerodrome_ai_agent.__version__ == "1.0.0"
    
    # Test component imports
    from aerodrome_ai_agent.brain import BrainConfig
    from aerodrome_ai_agent.memory import MemoryConfig
    from aerodrome_ai_agent.cdp import CDPConfig
    from aerodrome_ai_agent.config import AgentConfig
    
    # Verify classes can be instantiated
    brain_config = BrainConfig()
    memory_config = MemoryConfig()
    cdp_config = CDPConfig()
    agent_config = AgentConfig()
    
    assert brain_config.confidence_threshold == 0.7
    assert memory_config.user_id == "aerodrome_agent"
    assert cdp_config.network_id == "base-mainnet"
    assert agent_config.agent_name == "aerodrome-agent"


def test_configuration():
    """Test configuration loading and validation"""
    
    from aerodrome_ai_agent.config import AgentConfig, Settings
    from aerodrome_ai_agent.brain.config import BrainConfig
    
    # Test default configuration
    config = AgentConfig()
    config.validate()  # Should not raise
    
    # Test brain config validation
    brain_config = BrainConfig()
    brain_config.validate()  # Should not raise
    
    # Test invalid configuration
    brain_config.confidence_threshold = 1.5  # Invalid value
    with pytest.raises(ValueError):
        brain_config.validate()


def test_state_creation():
    """Test brain state creation"""
    
    from aerodrome_ai_agent.brain.state import create_initial_state, BrainState
    
    state = create_initial_state()
    
    # Verify required fields exist
    assert "timestamp" in state
    assert "cycle_count" in state
    assert "market_data" in state
    assert "wallet_balance" in state
    assert "opportunities" in state
    assert "execution_status" in state
    
    # Verify initial values
    assert state["cycle_count"] == 0
    assert state["execution_status"] == "idle"
    assert state["emergency_stop_active"] is False


@pytest.mark.asyncio
async def test_memory_system():
    """Test memory system basic functionality"""
    
    from aerodrome_ai_agent.memory import MemorySystem, MemoryConfig
    
    # Create memory system with test config
    config = MemoryConfig()
    memory = MemorySystem(config)
    
    # Test memory stats
    stats = await memory.get_memory_stats()
    assert "total_memories" in stats
    assert stats["total_memories"] == 0  # Should be empty initially


if __name__ == "__main__":
    # Run basic import test
    test_imports()
    print("✅ All basic tests passed!")