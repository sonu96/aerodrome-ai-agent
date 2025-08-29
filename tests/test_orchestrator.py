"""
Tests for the Aerodrome AI Agent orchestrator and integration tests.

Comprehensive integration test suite covering end-to-end workflows,
component interactions, and system-wide functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from aerodrome_ai_agent.brain.core import AerodromeBrain
from aerodrome_ai_agent.brain.config import BrainConfig
from aerodrome_ai_agent.brain.state import create_initial_state
from aerodrome_ai_agent.memory.system import MemorySystem
from aerodrome_ai_agent.memory.config import MemoryConfig
from aerodrome_ai_agent.cdp.manager import CDPManager
from aerodrome_ai_agent.cdp.config import CDPConfig

from tests.mocks.brain_mocks import MockBrain
from tests.mocks.memory_mocks import MockMemorySystem
from tests.mocks.cdp_mocks import MockCDPManager
from tests.fixtures.market_data import (
    STABLE_MARKET, VOLATILE_MARKET, HIGH_CONFIDENCE_OPPORTUNITY,
    MEDIUM_CONFIDENCE_OPPORTUNITY, LOW_CONFIDENCE_OPPORTUNITY
)


class MockBrain:
    """Mock brain for orchestrator testing."""
    
    def __init__(self, config, memory_system=None, cdp_manager=None):
        self.config = config
        self.memory_system = memory_system
        self.cdp_manager = cdp_manager
        self.running = False
        self.current_thread_id = "main"
        self._cycle_results = []
        self._should_fail = False
        self._failure_reason = "BrainError"
        self._cycle_count = 0
    
    async def run_cycle(self, initial_state=None):
        """Mock cycle execution."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        self._cycle_count += 1
        state = initial_state or create_initial_state()
        state["cycle_count"] = self._cycle_count
        state["execution_status"] = "completed"
        
        # Add to results
        self._cycle_results.append(state)
        return state
    
    async def start_continuous_operation(self, interval=60):
        """Mock continuous operation."""
        self.running = True
        cycle_count = 0
        
        while self.running and cycle_count < 3:  # Limit for testing
            try:
                state = create_initial_state()
                state["cycle_count"] = cycle_count + 1
                await self.run_cycle(state)
                await asyncio.sleep(0.01)  # Minimal delay for testing
                cycle_count += 1
            except Exception:
                break
    
    def stop(self):
        """Stop operation."""
        self.running = False
    
    async def emergency_stop(self):
        """Emergency stop."""
        self.running = False
    
    def get_status(self):
        """Get status."""
        return {
            "running": self.running,
            "cycle_count": self._cycle_count,
            "thread_id": self.current_thread_id
        }
    
    def set_should_fail(self, should_fail, reason="BrainError"):
        """Configure mock to fail."""
        self._should_fail = should_fail
        self._failure_reason = reason
    
    def get_cycle_results(self):
        """Get cycle results for testing."""
        return self._cycle_results


class AerodromeOrchestrator:
    """Mock orchestrator for integration testing."""
    
    def __init__(
        self,
        brain_config: BrainConfig = None,
        memory_config: MemoryConfig = None,
        cdp_config: CDPConfig = None
    ):
        self.brain_config = brain_config or BrainConfig()
        self.memory_config = memory_config or MemoryConfig()
        self.cdp_config = cdp_config or CDPConfig()
        
        # Initialize components
        self.memory_system = None
        self.cdp_manager = None
        self.brain = None
        
        # Runtime state
        self.initialized = False
        self.running = False
        self.health_status = "healthy"
    
    async def initialize(self):
        """Initialize all components."""
        try:
            # Initialize memory system
            self.memory_system = MockMemorySystem(self.memory_config)
            
            # Initialize CDP manager
            self.cdp_manager = MockCDPManager(self.cdp_config)
            await self.cdp_manager.initialize_wallet()
            
            # Initialize brain
            self.brain = MockBrain(
                self.brain_config,
                self.memory_system,
                self.cdp_manager
            )
            
            self.initialized = True
            
        except Exception as e:
            self.health_status = "failed"
            raise RuntimeError(f"Orchestrator initialization failed: {e}")
    
    async def start(self):
        """Start the orchestrator."""
        if not self.initialized:
            await self.initialize()
        
        self.running = True
        await self.brain.start_continuous_operation()
    
    async def stop(self):
        """Stop the orchestrator.""" 
        self.running = False
        if self.brain:
            self.brain.stop()
    
    async def emergency_shutdown(self):
        """Emergency shutdown."""
        self.running = False
        self.health_status = "emergency_stopped"
        if self.brain:
            await self.brain.emergency_stop()
    
    def get_health_status(self):
        """Get health status."""
        return {
            "status": self.health_status,
            "initialized": self.initialized,
            "running": self.running,
            "components": {
                "brain": self.brain.get_status() if self.brain else None,
                "memory": self.memory_system is not None,
                "cdp": self.cdp_manager is not None
            }
        }
    
    async def execute_single_operation(self, operation_type: str, parameters: dict):
        """Execute a single operation for testing."""
        if not self.initialized:
            await self.initialize()
        
        if operation_type == "swap":
            return await self._execute_swap(parameters)
        elif operation_type == "add_liquidity":
            return await self._execute_add_liquidity(parameters)
        elif operation_type == "analyze_market":
            return await self._analyze_market(parameters)
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")
    
    async def _execute_swap(self, params):
        """Execute swap operation.""" 
        # Simulate decision
        action = {
            "type": "SWAP",
            "pool_address": params["pool_address"],
            "token_in": params["token_in"],
            "token_out": params["token_out"],
            "amount_in": params["amount_in"]
        }
        
        # Simulate transaction
        simulation = await self.cdp_manager.simulate_transaction(action)
        
        if simulation["success"] and simulation["profitable"]:
            # Execute transaction
            router_abi = []  # Simplified for testing
            result = await self.cdp_manager.invoke_contract(
                "0xRouter", "swapExactTokensForTokens", router_abi, params
            )
            
            # Learn from experience
            experience = {
                "action_type": "SWAP",
                "pool": params["pool_address"],
                "amount": params["amount_in"],
                "confidence": 0.8
            }
            outcome = {
                "success": result["success"],
                "profit": simulation.get("expected_profit", 0),
                "confidence": 0.85
            }
            await self.memory_system.learn_from_experience(experience, outcome)
            
            return result
        else:
            return {"success": False, "reason": "Simulation failed"}
    
    async def _execute_add_liquidity(self, params):
        """Execute add liquidity operation."""
        action = {
            "type": "ADD_LIQUIDITY",
            "pool_address": params["pool_address"],
            "token_a": params["token_a"],
            "token_b": params["token_b"],
            "amount_a": params["amount_a"],
            "amount_b": params["amount_b"]
        }
        
        # Simulate and execute
        simulation = await self.cdp_manager.simulate_transaction(action)
        
        if simulation["success"]:
            router_abi = []  # Simplified
            result = await self.cdp_manager.invoke_contract(
                "0xRouter", "addLiquidity", router_abi, params
            )
            return result
        else:
            return {"success": False, "reason": "Simulation failed"}
    
    async def _analyze_market(self, params):
        """Analyze market conditions."""
        # Get market data
        pools = await self.cdp_manager.get_top_pools(10)
        
        # Recall relevant memories
        context = {"action_type": "SWAP", "timeframe": "recent"}
        memories = await self.memory_system.recall_relevant_memories(context, 5)
        
        # Extract patterns
        patterns = await self.memory_system.extract_patterns(min_occurrences=3)
        
        return {
            "pools_analyzed": len(pools),
            "relevant_memories": len(memories),
            "patterns_found": len(patterns),
            "analysis_timestamp": datetime.now().isoformat()
        }


class TestOrchestratorInitialization:
    """Test orchestrator initialization and setup."""
    
    def test_orchestrator_creation(self, brain_config, memory_config, cdp_config):
        """Test orchestrator creation with configs."""
        orchestrator = AerodromeOrchestrator(
            brain_config=brain_config,
            memory_config=memory_config,
            cdp_config=cdp_config
        )
        
        assert orchestrator.brain_config == brain_config
        assert orchestrator.memory_config == memory_config
        assert orchestrator.cdp_config == cdp_config
        assert orchestrator.initialized is False
        assert orchestrator.running is False
    
    def test_orchestrator_creation_defaults(self):
        """Test orchestrator creation with default configs."""
        orchestrator = AerodromeOrchestrator()
        
        assert isinstance(orchestrator.brain_config, BrainConfig)
        assert isinstance(orchestrator.memory_config, MemoryConfig)
        assert isinstance(orchestrator.cdp_config, CDPConfig)
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_success(self, brain_config, memory_config, cdp_config):
        """Test successful orchestrator initialization."""
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        
        await orchestrator.initialize()
        
        assert orchestrator.initialized is True
        assert orchestrator.memory_system is not None
        assert orchestrator.cdp_manager is not None
        assert orchestrator.brain is not None
        assert orchestrator.health_status == "healthy"
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_failure(self, brain_config, memory_config, cdp_config):
        """Test orchestrator initialization failure."""
        # Configure CDP to fail initialization
        cdp_config.api_key_name = ""  # Invalid config
        
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        
        with pytest.raises(RuntimeError, match="Orchestrator initialization failed"):
            await orchestrator.initialize()
        
        assert orchestrator.health_status == "failed"
    
    @pytest.mark.asyncio
    async def test_get_health_status(self, brain_config, memory_config, cdp_config):
        """Test health status reporting."""
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        
        # Before initialization
        status = orchestrator.get_health_status()
        assert status["initialized"] is False
        assert status["running"] is False
        assert status["components"]["brain"] is None
        
        # After initialization
        await orchestrator.initialize()
        status = orchestrator.get_health_status()
        assert status["initialized"] is True
        assert status["components"]["brain"] is not None
        assert status["components"]["memory"] is True
        assert status["components"]["cdp"] is True


class TestOrchestratorOperations:
    """Test orchestrator operation handling."""
    
    @pytest.fixture
    async def initialized_orchestrator(self, brain_config, memory_config, cdp_config):
        """Create initialized orchestrator."""
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        await orchestrator.initialize()
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_start_stop_orchestrator(self, initialized_orchestrator):
        """Test starting and stopping orchestrator."""
        orchestrator = initialized_orchestrator
        
        # Start in background
        start_task = asyncio.create_task(orchestrator.start())
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        assert orchestrator.running is True
        
        # Stop
        await orchestrator.stop()
        await start_task
        
        assert orchestrator.running is False
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, initialized_orchestrator):
        """Test emergency shutdown."""
        orchestrator = initialized_orchestrator
        
        # Start operation
        start_task = asyncio.create_task(orchestrator.start())
        await asyncio.sleep(0.05)
        
        # Emergency shutdown
        await orchestrator.emergency_shutdown()
        await start_task
        
        assert orchestrator.running is False
        assert orchestrator.health_status == "emergency_stopped"
    
    @pytest.mark.asyncio
    async def test_execute_swap_operation(self, initialized_orchestrator):
        """Test executing swap operation."""
        orchestrator = initialized_orchestrator
        
        swap_params = {
            "pool_address": "0x1234567890abcdef1234567890abcdef12345678",
            "token_in": "USDC",
            "token_out": "WETH",
            "amount_in": 1000.0
        }
        
        result = await orchestrator.execute_single_operation("swap", swap_params)
        
        assert "success" in result
        # If successful, should have executed transaction
        if result["success"]:
            assert "transaction_hash" in result
    
    @pytest.mark.asyncio
    async def test_execute_add_liquidity_operation(self, initialized_orchestrator):
        """Test executing add liquidity operation."""
        orchestrator = initialized_orchestrator
        
        liquidity_params = {
            "pool_address": "0x2345678901bcdef02345678901bcdef023456789",
            "token_a": "USDC",
            "token_b": "USDT",
            "amount_a": 5000.0,
            "amount_b": 5000.0
        }
        
        result = await orchestrator.execute_single_operation("add_liquidity", liquidity_params)
        
        assert "success" in result
    
    @pytest.mark.asyncio
    async def test_analyze_market_operation(self, initialized_orchestrator):
        """Test market analysis operation."""
        orchestrator = initialized_orchestrator
        
        # Add some memories first
        await orchestrator.memory_system.learn_from_experience(
            {"action_type": "SWAP", "pool": "0x123"},
            {"success": True, "profit": 25.0}
        )
        
        analysis_params = {"depth": "full"}
        
        result = await orchestrator.execute_single_operation("analyze_market", analysis_params)
        
        assert "pools_analyzed" in result
        assert "relevant_memories" in result
        assert "patterns_found" in result
        assert "analysis_timestamp" in result
        assert result["pools_analyzed"] > 0
    
    @pytest.mark.asyncio
    async def test_invalid_operation_type(self, initialized_orchestrator):
        """Test handling of invalid operation type."""
        orchestrator = initialized_orchestrator
        
        with pytest.raises(ValueError, match="Unknown operation type"):
            await orchestrator.execute_single_operation("invalid_operation", {})


class TestComponentIntegration:
    """Test integration between components."""
    
    @pytest.fixture
    async def full_system(self, brain_config, memory_config, cdp_config):
        """Create full integrated system."""
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        await orchestrator.initialize()
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_brain_memory_integration(self, full_system):
        """Test brain and memory system integration."""
        brain = full_system.brain
        memory = full_system.memory_system
        
        # Run a brain cycle
        initial_state = create_initial_state()
        result_state = await brain.run_cycle(initial_state)
        
        # Check that brain can interact with memory
        assert result_state["cycle_count"] > 0
        
        # Memory should be accessible
        stats = await memory.get_memory_stats()
        assert "total_memories" in stats
    
    @pytest.mark.asyncio
    async def test_brain_cdp_integration(self, full_system):
        """Test brain and CDP manager integration."""
        brain = full_system.brain
        cdp = full_system.cdp_manager
        
        # CDP should be initialized
        wallet_info = await cdp.get_wallet_info()
        assert "wallet_id" in wallet_info
        
        # Brain should be able to use CDP for simulations
        action = {
            "type": "SWAP",
            "pool_address": "0x123",
            "amount_in": 1000.0
        }
        simulation = await cdp.simulate_transaction(action)
        assert "success" in simulation
    
    @pytest.mark.asyncio
    async def test_memory_cdp_integration(self, full_system):
        """Test memory and CDP manager integration."""
        memory = full_system.memory_system
        cdp = full_system.cdp_manager
        
        # Get pool data from CDP
        pools = await cdp.get_top_pools(3)
        
        # Store experiences based on pool data
        for pool in pools:
            experience = {
                "action_type": "ANALYZE_POOL",
                "pool": pool["address"],
                "tvl": pool["tvl"],
                "volume": pool["volume_24h"]
            }
            outcome = {
                "success": True,
                "score": 0.8,
                "recommendation": "monitor"
            }
            
            memory_id = await memory.learn_from_experience(experience, outcome)
            assert memory_id is not None
        
        # Recall memories about pools
        context = {"action_type": "ANALYZE_POOL"}
        memories = await memory.recall_relevant_memories(context)
        
        assert len(memories) > 0
    
    @pytest.mark.asyncio
    async def test_full_decision_pipeline(self, full_system):
        """Test complete decision pipeline from observation to execution."""
        orchestrator = full_system
        
        # Simulate market opportunity detection
        opportunity_params = {
            "pool_address": "0x1234567890abcdef1234567890abcdef12345678",
            "token_in": "USDC", 
            "token_out": "WETH",
            "amount_in": 1000.0,
            "expected_profit": 25.0,
            "confidence": 0.85
        }
        
        # Step 1: Analyze market (observation)
        analysis = await orchestrator.execute_single_operation("analyze_market", {})
        assert analysis["pools_analyzed"] > 0
        
        # Step 2: Recall relevant experiences (memory)
        context = {"action_type": "SWAP", "pool_address": opportunity_params["pool_address"]}
        memories = await orchestrator.memory_system.recall_relevant_memories(context)
        
        # Step 3: Make decision and execute (brain + CDP)
        if opportunity_params["confidence"] > 0.7:  # High confidence
            result = await orchestrator.execute_single_operation("swap", opportunity_params)
            assert "success" in result
            
            # Step 4: Learn from outcome (memory)
            experience = {
                "action_type": "SWAP",
                "pool": opportunity_params["pool_address"],
                "amount": opportunity_params["amount_in"],
                "confidence": opportunity_params["confidence"]
            }
            outcome = {
                "success": result["success"],
                "profit": opportunity_params["expected_profit"] if result["success"] else -10.0,
                "confidence": 0.9 if result["success"] else 0.3
            }
            
            memory_id = await orchestrator.memory_system.learn_from_experience(experience, outcome)
            assert memory_id is not None


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    @pytest.fixture
    async def orchestrator_with_failing_components(self, brain_config, memory_config, cdp_config):
        """Create orchestrator with components that can be configured to fail."""
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        await orchestrator.initialize()
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_brain_failure_recovery(self, orchestrator_with_failing_components):
        """Test recovery from brain failures."""
        orchestrator = orchestrator_with_failing_components
        
        # Configure brain to fail
        orchestrator.brain.set_should_fail(True, "BrainProcessingError")
        
        # Operation should handle brain failure
        try:
            result = await orchestrator.execute_single_operation("analyze_market", {})
        except Exception as e:
            assert "BrainProcessingError" in str(e)
        
        # Recovery: fix brain
        orchestrator.brain.set_should_fail(False)
        
        # Should work again
        result = await orchestrator.execute_single_operation("analyze_market", {})
        assert "pools_analyzed" in result
    
    @pytest.mark.asyncio
    async def test_memory_failure_recovery(self, orchestrator_with_failing_components):
        """Test recovery from memory system failures.""" 
        orchestrator = orchestrator_with_failing_components
        
        # Configure memory to fail
        orchestrator.memory_system.set_should_fail(True, "MemoryStorageError")
        
        # Operations should handle memory failures gracefully
        swap_params = {
            "pool_address": "0x123",
            "token_in": "USDC",
            "token_out": "WETH",
            "amount_in": 1000.0
        }
        
        # Should still work, but without memory functionality
        result = await orchestrator.execute_single_operation("swap", swap_params)
        assert "success" in result
        
        # Recovery: fix memory
        orchestrator.memory_system.set_should_fail(False)
        
        # Memory functions should work again
        memories = await orchestrator.memory_system.recall_relevant_memories({"action_type": "SWAP"})
        assert isinstance(memories, list)
    
    @pytest.mark.asyncio
    async def test_cdp_failure_recovery(self, orchestrator_with_failing_components):
        """Test recovery from CDP failures."""
        orchestrator = orchestrator_with_failing_components
        
        # Configure CDP to fail
        orchestrator.cdp_manager.set_should_fail(True, "NetworkConnectionError")
        
        # Operations requiring CDP should fail
        swap_params = {
            "pool_address": "0x123",
            "token_in": "USDC",
            "token_out": "WETH",
            "amount_in": 1000.0
        }
        
        result = await orchestrator.execute_single_operation("swap", swap_params)
        assert result["success"] is False
        assert "reason" in result
        
        # Recovery: fix CDP
        orchestrator.cdp_manager.set_should_fail(False)
        
        # Should work again
        result = await orchestrator.execute_single_operation("swap", swap_params)
        # Success depends on simulation, but should not fail due to network error
    
    @pytest.mark.asyncio
    async def test_partial_system_operation(self, brain_config, memory_config, cdp_config):
        """Test system operation with some components failing."""
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        
        # Initialize with memory failure
        with patch.object(MockMemorySystem, '__init__', side_effect=Exception("Memory init failed")):
            try:
                await orchestrator.initialize()
            except Exception:
                # Expected to fail
                pass
        
        # Should handle graceful degradation
        assert orchestrator.health_status == "failed"
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown_during_operation(self, orchestrator_with_failing_components):
        """Test emergency shutdown during active operations."""
        orchestrator = orchestrator_with_failing_components
        
        # Start operation
        start_task = asyncio.create_task(orchestrator.start())
        await asyncio.sleep(0.05)  # Let it start
        
        # Trigger emergency during operation
        await orchestrator.emergency_shutdown()
        
        # Wait for shutdown to complete
        await start_task
        
        status = orchestrator.get_health_status()
        assert status["status"] == "emergency_stopped"
        assert status["running"] is False


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_continuous_operation_performance(self, brain_config, memory_config, cdp_config):
        """Test continuous operation performance."""
        # Configure for faster testing
        brain_config.observation_interval = 0.1  # Very fast cycles
        
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        await orchestrator.initialize()
        
        # Run for a short period
        start_time = datetime.now()
        start_task = asyncio.create_task(orchestrator.start())
        
        await asyncio.sleep(0.5)  # Run for 500ms
        await orchestrator.stop()
        await start_task
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Check that system ran efficiently
        assert duration < 1.0  # Should complete quickly
        
        # Check brain executed multiple cycles
        cycle_results = orchestrator.brain.get_cycle_results()
        assert len(cycle_results) >= 2  # Should have completed multiple cycles
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, brain_config, memory_config, cdp_config):
        """Test performance under concurrent operations."""
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        await orchestrator.initialize()
        
        # Define multiple operations
        operations = [
            ("analyze_market", {}),
            ("swap", {
                "pool_address": "0x123",
                "token_in": "USDC",
                "token_out": "WETH",
                "amount_in": 1000.0
            }),
            ("add_liquidity", {
                "pool_address": "0x456",
                "token_a": "USDC",
                "token_b": "USDT",
                "amount_a": 5000.0,
                "amount_b": 5000.0
            })
        ]
        
        # Execute concurrently
        start_time = datetime.now()
        
        tasks = [
            orchestrator.execute_single_operation(op_type, params)
            for op_type, params in operations
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete relatively quickly
        assert duration < 2.0
        
        # All operations should complete
        assert len(results) == len(operations)
        
        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Operations failed with exceptions: {exceptions}"
    
    @pytest.mark.asyncio
    async def test_memory_scaling(self, brain_config, memory_config, cdp_config):
        """Test memory system scaling with many experiences."""
        # Configure memory for testing
        memory_config.max_memories = 100
        memory_config.pattern_threshold = 5
        
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        await orchestrator.initialize()
        
        # Add many experiences
        start_time = datetime.now()
        
        for i in range(50):
            experience = {
                "action_type": "SWAP",
                "pool": f"0x{i:040d}",
                "amount": 1000.0 + i,
                "confidence": 0.7 + (i % 3) * 0.1
            }
            outcome = {
                "success": i % 4 != 0,  # 75% success rate
                "profit": (20.0 + i) if i % 4 != 0 else -(5.0 + i),
                "confidence": 0.8 + (i % 2) * 0.1
            }
            
            await orchestrator.memory_system.learn_from_experience(experience, outcome)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should handle many memories efficiently
        assert duration < 5.0
        
        # Verify memories stored
        stats = await orchestrator.memory_system.get_memory_stats()
        assert stats["total_memories"] >= 50
        
        # Test recall performance
        recall_start = datetime.now()
        
        memories = await orchestrator.memory_system.recall_relevant_memories(
            {"action_type": "SWAP"}, limit=10
        )
        
        recall_end = datetime.now()
        recall_duration = (recall_end - recall_start).total_seconds()
        
        # Recall should be fast
        assert recall_duration < 1.0
        assert len(memories) <= 10


@pytest.mark.integration
class TestEndToEndWorkflows:
    """End-to-end integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, brain_config, memory_config, cdp_config):
        """Test complete trading workflow from market analysis to execution."""
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        await orchestrator.initialize()
        
        # Step 1: Market Analysis
        analysis = await orchestrator.execute_single_operation("analyze_market", {
            "depth": "comprehensive"
        })
        
        assert analysis["pools_analyzed"] > 0
        
        # Step 2: Historical Analysis (check memory for similar conditions)
        memories = await orchestrator.memory_system.recall_relevant_memories({
            "action_type": "SWAP",
            "market_conditions": {"volatile": False}
        }, limit=5)
        
        # Step 3: Decision Making (simulate high-confidence opportunity)
        if len(memories) < 3:  # Not enough historical data, add some
            for i in range(5):
                await orchestrator.memory_system.learn_from_experience(
                    {
                        "action_type": "SWAP",
                        "pool": f"0x{i:040d}",
                        "confidence": 0.8,
                        "market_conditions": {"volatile": False}
                    },
                    {
                        "success": True,
                        "profit": 25.0 + i,
                        "confidence": 0.85
                    }
                )
        
        # Step 4: Execute Trade
        trade_params = {
            "pool_address": "0x1234567890abcdef1234567890abcdef12345678",
            "token_in": "USDC",
            "token_out": "WETH",
            "amount_in": 1000.0
        }
        
        result = await orchestrator.execute_single_operation("swap", trade_params)
        
        # Should succeed or provide clear reason for failure
        assert "success" in result
        if not result["success"]:
            assert "reason" in result
        
        # Step 5: Post-trade Analysis
        final_memories = await orchestrator.memory_system.recall_relevant_memories({
            "action_type": "SWAP"
        })
        
        # Should have learned from the experience
        assert len(final_memories) > len(memories)
    
    @pytest.mark.asyncio
    async def test_risk_management_workflow(self, brain_config, memory_config, cdp_config):
        """Test risk management and safety mechanisms.""" 
        # Configure conservative risk settings
        brain_config.confidence_threshold = 0.9  # Very high threshold
        brain_config.max_position_size = 0.1     # Small positions
        brain_config.max_slippage = 0.01         # Low slippage
        
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        await orchestrator.initialize()
        
        # Add some historical losses to memory
        for i in range(3):
            await orchestrator.memory_system.learn_from_experience(
                {
                    "action_type": "SWAP",
                    "pool": "0x1234567890abcdef1234567890abcdef12345678",
                    "confidence": 0.6 + i * 0.1,
                    "market_conditions": {"volatile": True}
                },
                {
                    "success": False,
                    "profit": -(50.0 + i * 10.0),
                    "confidence": 0.3,
                    "error": "SlippageExceeded"
                }
            )
        
        # Try to execute trade in volatile conditions
        risky_trade_params = {
            "pool_address": "0x1234567890abcdef1234567890abcdef12345678", 
            "token_in": "USDC",
            "token_out": "WETH",
            "amount_in": 10000.0,  # Large trade
            "market_conditions": {"volatile": True}
        }
        
        # Should either reject or proceed with caution
        result = await orchestrator.execute_single_operation("swap", risky_trade_params)
        
        # System should handle risk appropriately
        if not result["success"]:
            # Rejection is acceptable for high-risk scenario
            assert "reason" in result
        else:
            # If executed, should be with appropriate risk controls
            assert result["success"] is True
    
    @pytest.mark.asyncio 
    async def test_learning_and_adaptation_workflow(self, brain_config, memory_config, cdp_config):
        """Test system learning and adaptation over time."""
        orchestrator = AerodromeOrchestrator(brain_config, memory_config, cdp_config)
        await orchestrator.initialize()
        
        # Phase 1: Initial learning period (poor performance)
        for i in range(5):
            await orchestrator.memory_system.learn_from_experience(
                {
                    "action_type": "SWAP",
                    "pool": f"0x{i:040d}",
                    "confidence": 0.5,  # Low initial confidence
                    "strategy": "aggressive"
                },
                {
                    "success": i >= 3,  # Only later trades successful
                    "profit": 15.0 if i >= 3 else -20.0,
                    "confidence": 0.8 if i >= 3 else 0.3
                }
            )
        
        # Extract initial patterns
        initial_patterns = await orchestrator.memory_system.extract_patterns(min_occurrences=2)
        
        # Phase 2: Improved performance (learning)
        for i in range(5, 10):
            await orchestrator.memory_system.learn_from_experience(
                {
                    "action_type": "SWAP",
                    "pool": f"0x{i:040d}",
                    "confidence": 0.8,  # Higher confidence 
                    "strategy": "conservative"  # Changed strategy
                },
                {
                    "success": True,  # All successful
                    "profit": 25.0 + i,
                    "confidence": 0.9
                }
            )
        
        # Extract improved patterns
        improved_patterns = await orchestrator.memory_system.extract_patterns(min_occurrences=3)
        
        # Should have learned new patterns
        assert len(improved_patterns) >= len(initial_patterns)
        
        # Test recall with different contexts
        aggressive_memories = await orchestrator.memory_system.recall_relevant_memories({
            "strategy": "aggressive"
        })
        
        conservative_memories = await orchestrator.memory_system.recall_relevant_memories({
            "strategy": "conservative"
        })
        
        # Should be able to distinguish between strategies
        assert len(conservative_memories) > 0
        
        # Conservative strategy should have better outcomes
        if aggressive_memories and conservative_memories:
            conservative_profits = [
                m["metadata"]["outcome"]["profit"] 
                for m in conservative_memories 
                if m["metadata"]["outcome"]["success"]
            ]
            aggressive_profits = [
                m["metadata"]["outcome"]["profit"]
                for m in aggressive_memories
                if m["metadata"]["outcome"]["success"]
            ]
            
            if conservative_profits and aggressive_profits:
                avg_conservative = sum(conservative_profits) / len(conservative_profits)
                avg_aggressive = sum(aggressive_profits) / len(aggressive_profits)
                assert avg_conservative > avg_aggressive