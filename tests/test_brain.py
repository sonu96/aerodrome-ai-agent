"""
Tests for the Aerodrome Brain module.

Comprehensive test suite covering brain initialization, state management,
node execution, decision routing, and error handling.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from aerodrome_ai_agent.brain.core import AerodromeBrain
from aerodrome_ai_agent.brain.config import BrainConfig
from aerodrome_ai_agent.brain.state import create_initial_state, BrainState
from tests.mocks.cdp_mocks import MockCDPManager
from tests.mocks.memory_mocks import MockMemorySystem
from tests.fixtures.market_data import (
    STABLE_MARKET, VOLATILE_MARKET, HIGH_CONFIDENCE_OPPORTUNITY,
    MEDIUM_CONFIDENCE_OPPORTUNITY, LOW_CONFIDENCE_OPPORTUNITY
)


class TestBrainConfig:
    """Test brain configuration handling."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BrainConfig()
        
        assert config.confidence_threshold == 0.7
        assert config.risk_threshold == 0.3
        assert config.observation_interval == 60
        assert config.max_position_size == 0.2
        assert config.max_slippage == 0.02
    
    def test_config_validation_valid(self):
        """Test valid configuration passes validation."""
        config = BrainConfig(
            confidence_threshold=0.8,
            risk_threshold=0.2,
            max_position_size=0.1
        )
        
        # Should not raise
        config.validate()
    
    def test_config_validation_invalid_confidence(self):
        """Test invalid confidence threshold fails validation."""
        config = BrainConfig(confidence_threshold=1.5)
        
        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            config.validate()
    
    def test_config_validation_invalid_risk(self):
        """Test invalid risk threshold fails validation."""
        config = BrainConfig(risk_threshold=-0.1)
        
        with pytest.raises(ValueError, match="risk_threshold must be between 0 and 1"):
            config.validate()
    
    def test_config_validation_invalid_position_size(self):
        """Test invalid position size fails validation."""
        config = BrainConfig(max_position_size=1.1)
        
        with pytest.raises(ValueError, match="max_position_size must be between 0 and 1"):
            config.validate()
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = BrainConfig(confidence_threshold=0.8, risk_threshold=0.2)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["confidence_threshold"] == 0.8
        assert config_dict["risk_threshold"] == 0.2
        assert "observation_interval" in config_dict
    
    def test_config_from_dict(self):
        """Test configuration from dictionary creation."""
        config_dict = {
            "confidence_threshold": 0.9,
            "risk_threshold": 0.1,
            "observation_interval": 30
        }
        
        config = BrainConfig.from_dict(config_dict)
        
        assert config.confidence_threshold == 0.9
        assert config.risk_threshold == 0.1
        assert config.observation_interval == 30


class TestBrainState:
    """Test brain state management."""
    
    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state()
        
        assert isinstance(state, dict)
        assert "timestamp" in state
        assert "cycle_count" in state
        assert "execution_status" in state
        assert "market_data" in state
        assert "wallet_balance" in state
        assert "opportunities" in state
        
        assert state["cycle_count"] == 0
        assert state["execution_status"] == "idle"
        assert state["emergency_stop_active"] is False
    
    def test_state_update(self):
        """Test state updates."""
        state = create_initial_state()
        
        # Update state
        state.update({
            "cycle_count": 5,
            "execution_status": "running",
            "confidence_score": 0.85
        })
        
        assert state["cycle_count"] == 5
        assert state["execution_status"] == "running"
        assert state["confidence_score"] == 0.85
    
    def test_state_validation(self, sample_brain_state):
        """Test state validation helper."""
        from tests.conftest import assert_state_valid
        
        # Should not raise
        assert_state_valid(sample_brain_state)
    
    def test_state_invalid_missing_field(self):
        """Test state validation with missing required field."""
        from tests.conftest import assert_state_valid
        
        invalid_state = {"timestamp": datetime.now()}
        
        with pytest.raises(AssertionError, match="State missing required field"):
            assert_state_valid(invalid_state)


class TestAerodromeBrain:
    """Test main brain functionality."""
    
    @pytest.fixture
    def mock_memory_system(self):
        """Create mock memory system."""
        return MockMemorySystem()
    
    @pytest.fixture
    def mock_cdp_manager(self):
        """Create mock CDP manager.""" 
        return MockCDPManager()
    
    @pytest.fixture
    def brain(self, brain_config, mock_memory_system, mock_cdp_manager):
        """Create brain instance for testing."""
        return AerodromeBrain(
            config=brain_config,
            memory_system=mock_memory_system,
            cdp_manager=mock_cdp_manager
        )
    
    def test_brain_initialization(self, brain_config):
        """Test brain initialization."""
        brain = AerodromeBrain(config=brain_config)
        
        assert brain.config == brain_config
        assert brain.compiled_graph is not None
        assert brain.running is False
        assert brain.current_thread_id == "main"
    
    def test_brain_initialization_default_config(self):
        """Test brain initialization with default config."""
        brain = AerodromeBrain()
        
        assert isinstance(brain.config, BrainConfig)
        assert brain.config.confidence_threshold == 0.7
    
    @pytest.mark.asyncio
    async def test_initialize_state(self, brain, initial_brain_state):
        """Test state initialization."""
        initial_state = initial_brain_state.copy()
        initial_state["cycle_count"] = 3
        
        result_state = await brain._initialize_state(initial_state)
        
        assert result_state["cycle_count"] == 4  # Incremented
        assert result_state["execution_status"] == "initializing"
        assert isinstance(result_state["timestamp"], datetime)
        assert result_state["errors"] == []
        assert result_state["opportunities"] == []
    
    @pytest.mark.asyncio
    async def test_simulate_execution_success(self, brain, sample_brain_state, mock_cdp_manager):
        """Test successful simulation."""
        # Configure successful simulation
        mock_cdp_manager.set_simulation_result(
            "SWAP", 
            "0x1234567890abcdef",
            {
                "success": True,
                "profitable": True,
                "gas_estimate": 125000,
                "expected_output": 0.398
            }
        )
        
        result_state = await brain._simulate_execution(sample_brain_state)
        
        assert result_state["simulation_result"]["success"] is True
        assert result_state["simulation_result"]["profitable"] is True
        assert result_state["simulation_result"]["gas_estimate"] == 125000
    
    @pytest.mark.asyncio
    async def test_simulate_execution_failure(self, brain, sample_brain_state, mock_cdp_manager):
        """Test failed simulation."""
        # Configure failed simulation
        mock_cdp_manager.set_should_fail(True, "InsufficientLiquidity")
        
        result_state = await brain._simulate_execution(sample_brain_state)
        
        assert result_state["simulation_result"]["success"] is False
        assert "error" in result_state["simulation_result"]
    
    @pytest.mark.asyncio
    async def test_simulate_execution_no_action(self, brain, initial_brain_state):
        """Test simulation with no selected action."""
        result_state = await brain._simulate_execution(initial_brain_state)
        
        assert result_state["simulation_result"]["success"] is False
        assert "No action to simulate" in result_state["simulation_result"]["error"]
    
    def test_route_decision_high_confidence(self, brain, sample_brain_state):
        """Test decision routing with high confidence."""
        sample_brain_state["confidence_score"] = 0.85
        sample_brain_state["selected_action"] = HIGH_CONFIDENCE_OPPORTUNITY
        
        route = brain._route_decision(sample_brain_state)
        
        assert route == "simulate"
    
    def test_route_decision_low_confidence(self, brain, sample_brain_state):
        """Test decision routing with low confidence."""
        sample_brain_state["confidence_score"] = 0.5
        sample_brain_state["selected_action"] = LOW_CONFIDENCE_OPPORTUNITY
        
        route = brain._route_decision(sample_brain_state)
        
        assert route == "skip"
    
    def test_route_decision_no_action(self, brain, sample_brain_state):
        """Test decision routing with no action selected."""
        sample_brain_state["selected_action"] = None
        
        route = brain._route_decision(sample_brain_state)
        
        assert route == "skip"
    
    def test_route_decision_emergency(self, brain, sample_brain_state):
        """Test decision routing with emergency stop active."""
        sample_brain_state["emergency_stop_active"] = True
        sample_brain_state["selected_action"] = HIGH_CONFIDENCE_OPPORTUNITY
        
        route = brain._route_decision(sample_brain_state)
        
        assert route == "emergency"
    
    def test_route_simulation_success(self, brain):
        """Test simulation routing with successful result."""
        state = {
            "simulation_result": {
                "success": True,
                "profitable": True
            }
        }
        
        route = brain._route_simulation(state)
        
        assert route == "execute"
    
    def test_route_simulation_failure(self, brain):
        """Test simulation routing with failed result."""
        state = {
            "simulation_result": {
                "success": False,
                "profitable": False
            }
        }
        
        route = brain._route_simulation(state)
        
        assert route == "reject"
    
    def test_route_simulation_unprofitable(self, brain):
        """Test simulation routing with unprofitable result."""
        state = {
            "simulation_result": {
                "success": True,
                "profitable": False
            }
        }
        
        route = brain._route_simulation(state)
        
        assert route == "retry"
    
    @pytest.mark.asyncio
    async def test_run_cycle_success(self, brain, initial_brain_state):
        """Test successful brain cycle execution.""" 
        # Mock the graph execution
        with patch.object(brain.compiled_graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_result = initial_brain_state.copy()
            mock_result["cycle_count"] = 1
            mock_result["execution_status"] = "completed"
            mock_invoke.return_value = mock_result
            
            result = await brain.run_cycle(initial_brain_state)
            
            assert result["cycle_count"] == 1
            assert result["execution_status"] == "completed"
            mock_invoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_cycle_failure(self, brain, initial_brain_state):
        """Test brain cycle with execution failure."""
        # Mock the graph to raise an exception
        with patch.object(brain.compiled_graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = Exception("Graph execution failed")
            
            result = await brain.run_cycle(initial_brain_state)
            
            assert len(result["errors"]) > 0
            assert "Graph execution failed" in str(result["errors"][0]["error"])
    
    @pytest.mark.asyncio
    async def test_run_cycle_default_state(self, brain):
        """Test brain cycle with default initial state."""
        with patch.object(brain.compiled_graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = create_initial_state()
            
            result = await brain.run_cycle()
            
            assert isinstance(result, dict)
            mock_invoke.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_start_continuous_operation(self, brain):
        """Test continuous brain operation startup."""
        with patch.object(brain, 'run_cycle', new_callable=AsyncMock) as mock_run_cycle:
            mock_run_cycle.return_value = create_initial_state()
            
            # Start operation in background
            task = asyncio.create_task(brain.start_continuous_operation(interval=0.1))
            
            # Let it run briefly
            await asyncio.sleep(0.25)
            
            # Stop operation
            brain.stop()
            await task
            
            # Should have executed multiple cycles
            assert mock_run_cycle.call_count >= 2
            assert brain.running is False
    
    @pytest.mark.asyncio
    async def test_start_continuous_operation_already_running(self, brain):
        """Test starting continuous operation when already running."""
        brain.running = True
        
        # Should return immediately without starting
        await brain.start_continuous_operation()
        
        # No assertions needed, just ensure no exception
    
    @pytest.mark.asyncio
    async def test_start_continuous_operation_with_error(self, brain):
        """Test continuous operation handling errors."""
        with patch.object(brain, 'run_cycle', new_callable=AsyncMock) as mock_run_cycle:
            # First call succeeds, second call fails, third succeeds
            mock_run_cycle.side_effect = [
                create_initial_state(),
                Exception("Temporary error"),
                create_initial_state()
            ]
            
            # Start operation
            task = asyncio.create_task(brain.start_continuous_operation(interval=0.1))
            
            # Let it run and handle the error
            await asyncio.sleep(0.35)
            
            brain.stop()
            await task
            
            # Should continue after error
            assert mock_run_cycle.call_count >= 3
    
    def test_stop(self, brain):
        """Test stopping brain operation."""
        brain.running = True
        
        brain.stop()
        
        assert brain.running is False
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self, brain):
        """Test emergency stop functionality."""
        brain.running = True
        
        await brain.emergency_stop()
        
        assert brain.running is False
    
    def test_get_status(self, brain):
        """Test status retrieval."""
        status = brain.get_status()
        
        assert isinstance(status, dict)
        assert "running" in status
        assert "thread_id" in status
        assert "config" in status
        assert "timestamp" in status
        assert status["running"] is False
        assert status["thread_id"] == "main"


class TestBrainIntegration:
    """Integration tests for brain components."""
    
    @pytest.fixture
    def integrated_brain(self, brain_config):
        """Create brain with real components for integration testing."""
        mock_memory = MockMemorySystem(brain_config)
        mock_cdp = MockCDPManager(brain_config)
        
        return AerodromeBrain(
            config=brain_config,
            memory_system=mock_memory,
            cdp_manager=mock_cdp
        )
    
    @pytest.mark.asyncio
    async def test_full_cycle_with_high_confidence_opportunity(self, integrated_brain):
        """Test complete cycle with high confidence opportunity."""
        # Create state with high confidence opportunity
        initial_state = create_initial_state()
        initial_state.update({
            "market_data": STABLE_MARKET,
            "opportunities": [HIGH_CONFIDENCE_OPPORTUNITY],
            "selected_action": HIGH_CONFIDENCE_OPPORTUNITY,
            "confidence_score": 0.9
        })
        
        with patch.object(integrated_brain.compiled_graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            # Simulate successful execution
            result_state = initial_state.copy()
            result_state.update({
                "cycle_count": 1,
                "execution_status": "completed",
                "execution_result": {"success": True, "profit": 25.50}
            })
            mock_invoke.return_value = result_state
            
            result = await integrated_brain.run_cycle(initial_state)
            
            assert result["execution_status"] == "completed"
            assert result["execution_result"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_full_cycle_with_low_confidence_opportunity(self, integrated_brain):
        """Test complete cycle with low confidence opportunity."""
        initial_state = create_initial_state()
        initial_state.update({
            "market_data": VOLATILE_MARKET,
            "opportunities": [LOW_CONFIDENCE_OPPORTUNITY],
            "selected_action": LOW_CONFIDENCE_OPPORTUNITY,
            "confidence_score": 0.5  # Below threshold
        })
        
        with patch.object(integrated_brain.compiled_graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            # Should skip execution due to low confidence
            result_state = initial_state.copy()
            result_state.update({
                "cycle_count": 1,
                "execution_status": "skipped",
                "decision_rationale": "Confidence below threshold"
            })
            mock_invoke.return_value = result_state
            
            result = await integrated_brain.run_cycle(initial_state)
            
            assert result["execution_status"] == "skipped"
    
    @pytest.mark.asyncio
    async def test_emergency_stop_integration(self, integrated_brain):
        """Test emergency stop during cycle execution."""
        initial_state = create_initial_state()
        initial_state["emergency_stop_active"] = True
        
        with patch.object(integrated_brain.compiled_graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            result_state = initial_state.copy()
            result_state.update({
                "cycle_count": 1,
                "execution_status": "emergency_stopped"
            })
            mock_invoke.return_value = result_state
            
            result = await integrated_brain.run_cycle(initial_state)
            
            assert result["execution_status"] == "emergency_stopped"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_learning_integration(self, integrated_brain):
        """Test integration with memory system for learning."""
        # Configure memory system
        memory_system = integrated_brain.memory_system
        
        # Add some experience
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
        
        memory_id = await memory_system.learn_from_experience(experience, outcome)
        
        # Verify memory was stored
        assert memory_id is not None
        assert memory_system.get_memory_count() > 0
        
        # Test recall
        context = {"action_type": "SWAP", "pool": "0x1234567890abcdef"}
        memories = await memory_system.recall_relevant_memories(context, limit=5)
        
        assert len(memories) > 0
        assert any(mem["metadata"]["experience"]["action_type"] == "SWAP" for mem in memories)


class TestBrainErrorHandling:
    """Test error handling in brain operations."""
    
    @pytest.fixture
    def brain_with_failing_components(self, brain_config):
        """Create brain with components configured to fail."""
        mock_memory = MockMemorySystem(brain_config)
        mock_memory.set_should_fail(True, "MemorySystemError")
        
        mock_cdp = MockCDPManager(brain_config)
        mock_cdp.set_should_fail(True, "CDPError")
        
        return AerodromeBrain(
            config=brain_config,
            memory_system=mock_memory,
            cdp_manager=mock_cdp
        )
    
    @pytest.mark.asyncio
    async def test_memory_system_failure(self, brain_with_failing_components):
        """Test handling of memory system failures."""
        # The brain should still be able to run without memory
        initial_state = create_initial_state()
        
        with patch.object(brain_with_failing_components.compiled_graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            # Simulate graph execution with memory error handling
            result_state = initial_state.copy()
            result_state["cycle_count"] = 1
            result_state["warnings"] = ["Memory system unavailable"]
            mock_invoke.return_value = result_state
            
            result = await brain_with_failing_components.run_cycle(initial_state)
            
            assert result["cycle_count"] == 1
            assert len(result.get("warnings", [])) > 0
    
    @pytest.mark.asyncio
    async def test_cdp_manager_failure(self, brain_with_failing_components):
        """Test handling of CDP manager failures."""
        initial_state = create_initial_state()
        initial_state.update({
            "selected_action": HIGH_CONFIDENCE_OPPORTUNITY,
            "confidence_score": 0.9
        })
        
        with patch.object(brain_with_failing_components.compiled_graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            result_state = initial_state.copy()
            result_state["cycle_count"] = 1
            result_state["execution_status"] = "failed"
            result_state["errors"] = [{"error": "CDPError", "node": "execution"}]
            mock_invoke.return_value = result_state
            
            result = await brain_with_failing_components.run_cycle(initial_state)
            
            assert result["execution_status"] == "failed"
            assert len(result["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_configuration_validation_failure(self):
        """Test brain initialization with invalid configuration."""
        invalid_config = BrainConfig(confidence_threshold=1.5)  # Invalid value
        
        with pytest.raises(ValueError):
            AerodromeBrain(config=invalid_config)
    
    @pytest.mark.asyncio
    async def test_graph_compilation_failure(self, brain_config):
        """Test handling of graph compilation failures."""
        with patch('aerodrome_ai_agent.brain.core.StateGraph') as mock_state_graph:
            mock_graph = MagicMock()
            mock_graph.compile.side_effect = Exception("Graph compilation failed")
            mock_state_graph.return_value = mock_graph
            
            with pytest.raises(Exception, match="Graph compilation failed"):
                AerodromeBrain(config=brain_config)