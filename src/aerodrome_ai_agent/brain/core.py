"""
Aerodrome Brain Core - Main LangGraph-based cognitive system

The brain operates as a directed graph with nodes representing cognitive functions:
- Initialize: Set up initial state
- Observe: Market data collection  
- Recall: Memory retrieval
- Analyze: Opportunity analysis
- Decide: Risk-adjusted decisions
- Execute: Blockchain operations
- Learn: Pattern extraction
- Monitor: Performance tracking
"""

from typing import Dict, Any, Optional, Callable
import asyncio
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .config import BrainConfig
from .state import BrainState, create_initial_state
from .nodes import (
    ObservationNode,
    MemoryNode, 
    AnalysisNode,
    DecisionNode,
    ExecutionNode,
    LearningNode,
    MonitoringNode
)
from ..memory import MemorySystem
from ..cdp import CDPManager


logger = logging.getLogger(__name__)


class AerodromeBrain:
    """
    Main brain class implementing LangGraph state machine for autonomous DeFi operations
    """
    
    def __init__(
        self,
        config: Optional[BrainConfig] = None,
        memory_system: Optional[MemorySystem] = None,
        cdp_manager: Optional[CDPManager] = None
    ):
        self.config = config or BrainConfig()
        self.config.validate()
        
        self.memory_system = memory_system
        self.cdp_manager = cdp_manager
        
        # Initialize nodes
        self.observation_node = ObservationNode(cdp_manager, self.config)
        self.memory_node = MemoryNode(memory_system, self.config)
        self.analysis_node = AnalysisNode(self.config)
        self.decision_node = DecisionNode(self.config)
        self.execution_node = ExecutionNode(cdp_manager, self.config)
        self.learning_node = LearningNode(memory_system, self.config)
        self.monitoring_node = MonitoringNode(self.config)
        
        # Build and compile graph
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile(
            checkpointer=MemorySaver(),
            interrupt_before=[],
            interrupt_after=[]
        )
        
        # Runtime state
        self.running = False
        self.current_thread_id = "main"
        
        logger.info("Aerodrome Brain initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Construct the brain's cognitive graph"""
        
        graph = StateGraph(BrainState)
        
        # Add all cognitive nodes
        graph.add_node("initialize", self._initialize_state)
        graph.add_node("observe", self.observation_node.execute)
        graph.add_node("recall", self.memory_node.execute)
        graph.add_node("analyze", self.analysis_node.execute)
        graph.add_node("decide", self.decision_node.execute)
        graph.add_node("simulate", self._simulate_execution)
        graph.add_node("execute", self.execution_node.execute)
        graph.add_node("monitor", self.monitoring_node.execute)
        graph.add_node("learn", self.learning_node.execute)
        graph.add_node("update_memory", self.memory_node.update)
        graph.add_node("prune", self.memory_node.prune)
        
        # Define the flow
        graph.set_entry_point("initialize")
        
        # Linear flow through cognitive pipeline
        graph.add_edge("initialize", "observe")
        graph.add_edge("observe", "recall")
        graph.add_edge("recall", "analyze")
        graph.add_edge("analyze", "decide")
        
        # Conditional routing from decision node
        graph.add_conditional_edges(
            "decide",
            self._route_decision,
            {
                "simulate": "simulate",
                "skip": "update_memory",
                "emergency": "monitor"
            }
        )
        
        # Conditional routing from simulation
        graph.add_conditional_edges(
            "simulate", 
            self._route_simulation,
            {
                "execute": "execute",
                "reject": "update_memory", 
                "retry": "analyze"
            }
        )
        
        # Execution flow
        graph.add_edge("execute", "monitor")
        graph.add_edge("monitor", "learn")
        graph.add_edge("learn", "update_memory")
        graph.add_edge("update_memory", "prune")
        graph.add_edge("prune", END)
        
        return graph
    
    async def _initialize_state(self, state: BrainState) -> BrainState:
        """Initialize the brain state for a new cycle"""
        
        # Reset cycle-specific data
        state["timestamp"] = datetime.now()
        state["cycle_count"] = state.get("cycle_count", 0) + 1
        state["execution_status"] = "initializing"
        state["errors"] = []
        state["warnings"] = []
        
        # Clear previous cycle data
        state["opportunities"] = []
        state["selected_action"] = None
        state["confidence_score"] = 0.0
        state["transaction_params"] = None
        state["simulation_result"] = None
        state["execution_result"] = None
        state["decision_rationale"] = ""
        
        logger.info(f"Brain cycle {state['cycle_count']} initialized")
        return state
    
    def _route_decision(self, state: BrainState) -> str:
        """Routing logic for decision node output"""
        
        # Check for emergency conditions
        if state.get("emergency_stop_active", False):
            return "emergency"
        
        # No action selected
        if state.get("selected_action") is None:
            return "skip"
        
        # Check confidence threshold  
        confidence = state.get("confidence_score", 0.0)
        if confidence >= self.config.confidence_threshold:
            return "simulate"
        
        # Low confidence, skip execution
        logger.info(f"Decision rejected: confidence {confidence} < {self.config.confidence_threshold}")
        return "skip"
    
    def _route_simulation(self, state: BrainState) -> str:
        """Routing logic for simulation results"""
        
        sim_result = state.get("simulation_result", {})
        
        if not sim_result.get("success", False):
            logger.warning("Simulation failed, rejecting action")
            return "reject"
        
        # Check profitability
        if not sim_result.get("profitable", False):
            logger.info("Simulation unprofitable, retrying with different parameters")
            return "retry"
        
        # Simulation successful, proceed to execution
        logger.info("Simulation successful, proceeding to execution")
        return "execute"
    
    async def _simulate_execution(self, state: BrainState) -> BrainState:
        """Simulate execution before actual transaction"""
        
        action = state.get("selected_action")
        if not action:
            state["simulation_result"] = {"success": False, "error": "No action to simulate"}
            return state
        
        try:
            # Use CDP SDK to simulate transaction
            if self.cdp_manager:
                sim_result = await self.cdp_manager.simulate_transaction(
                    action, 
                    state.get("transaction_params", {})
                )
            else:
                # Mock simulation for testing
                sim_result = {
                    "success": True,
                    "profitable": True,
                    "gas_estimate": 150000,
                    "expected_output": action.get("expected_output", 0)
                }
            
            state["simulation_result"] = sim_result
            logger.info("Transaction simulation completed successfully")
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            state["simulation_result"] = {
                "success": False,
                "error": str(e)
            }
        
        return state
    
    async def run_cycle(self, initial_state: Optional[BrainState] = None) -> BrainState:
        """Run a single brain cycle"""
        
        if initial_state is None:
            initial_state = create_initial_state()
        
        try:
            # Execute the graph
            result = await self.compiled_graph.ainvoke(
                initial_state,
                config={"thread_id": self.current_thread_id}
            )
            
            logger.info(f"Brain cycle {result.get('cycle_count', 0)} completed")
            return result
            
        except Exception as e:
            logger.error(f"Brain cycle failed: {e}")
            # Return state with error
            initial_state["errors"].append({
                "error": str(e),
                "timestamp": datetime.now(),
                "node": "unknown"
            })
            return initial_state
    
    async def start_continuous_operation(self, interval: Optional[int] = None):
        """Start continuous brain operation"""
        
        if self.running:
            logger.warning("Brain is already running")
            return
        
        interval = interval or self.config.observation_interval
        self.running = True
        
        logger.info(f"Starting continuous brain operation (interval: {interval}s)")
        
        state = create_initial_state()
        
        while self.running:
            try:
                # Run cycle
                state = await self.run_cycle(state)
                
                # Wait for next cycle
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Brain operation interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in brain operation: {e}")
                await asyncio.sleep(interval)
    
    def stop(self):
        """Stop continuous brain operation"""
        self.running = False
        logger.info("Brain operation stopped")
    
    async def emergency_stop(self):
        """Trigger emergency stop procedures"""
        logger.critical("Emergency stop activated!")
        self.running = False
        
        # Additional emergency procedures would go here
        # - Cancel pending transactions
        # - Close risky positions
        # - Alert administrators
        
    def get_status(self) -> Dict[str, Any]:
        """Get current brain status"""
        return {
            "running": self.running,
            "thread_id": self.current_thread_id,
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat()
        }