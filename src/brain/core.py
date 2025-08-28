"""
Core Brain implementation using LangGraph.

This module contains the main AerodromeBrain class that implements the cognitive
state machine using LangGraph, following current best practices for async operation,
state management, and graph construction.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Literal, Union

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from .state import BrainState, BrainConfig
from .nodes.observe import ObserverNode
from .nodes.recall import RecallNode
from .nodes.analyze import AnalyzeNode
from .nodes.decide import DecisionNode
from .nodes.execute import ExecutionNode
from .nodes.learn import LearningNode
from .algorithms import OpportunityScorer, RiskAssessor
from .errors import BrainErrorHandler


class AerodromeBrain:
    """
    Main brain class implementing LangGraph-based cognitive state machine.
    
    The brain operates as a continuous cycle:
    1. Observe market conditions
    2. Recall relevant memories and patterns
    3. Analyze opportunities 
    4. Make decisions based on risk assessment
    5. Execute actions via CDP SDK
    6. Learn from results and update memory
    
    Uses modern LangGraph patterns including:
    - Command objects for state updates and routing
    - Async node execution for performance
    - Conditional edges for dynamic flow control
    - State reducers for proper state management
    - Error recovery strategies
    """

    def __init__(
        self, 
        config: Optional[BrainConfig] = None,
        cdp_manager=None,
        memory_system=None
    ):
        """
        Initialize the Aerodrome Brain.
        
        Args:
            config: Brain configuration parameters
            cdp_manager: CDP SDK manager instance
            memory_system: Memory system for storing experiences
        """
        self.config = config or BrainConfig()
        self.cdp_manager = cdp_manager
        self.memory_system = memory_system
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize components
        self._initialize_components()
        
        # Build and compile the graph
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile(
            checkpointer=MemorySaver(),
            debug=True
        )
        
        # Runtime state
        self.is_running = False
        self.emergency_stop_active = False
        self.cycle_count = 0
        
        self.logger.info("AerodromeBrain initialized successfully")

    def _initialize_components(self):
        """Initialize all brain components and algorithms."""
        
        # Initialize node components
        self.observer = ObserverNode(self.cdp_manager, self.config)
        self.recall = RecallNode(self.memory_system, self.config)
        self.analyzer = AnalyzeNode(self.config)
        self.decision_maker = DecisionNode(self.config)
        self.executor = ExecutionNode(self.cdp_manager, self.config)
        self.learner = LearningNode(self.memory_system, self.config)
        
        # Initialize algorithms
        self.opportunity_scorer = OpportunityScorer(self.config)
        self.risk_assessor = RiskAssessor(self.config)
        
        # Initialize error handler
        self.error_handler = BrainErrorHandler(self.config)

    def _build_graph(self) -> StateGraph:
        """
        Build the cognitive state machine using LangGraph.
        
        Following LangGraph best practices:
        - Uses modern StateGraph construction
        - Implements proper async node patterns
        - Uses Command objects for routing and state updates
        - Includes conditional edges for dynamic flow
        """
        
        # Initialize graph with state schema
        graph = StateGraph(BrainState)
        
        # Add all cognitive nodes
        graph.add_node("initialize", self._initialize_cycle)
        graph.add_node("observe", self._observe_wrapper)
        graph.add_node("recall", self._recall_wrapper)
        graph.add_node("analyze", self._analyze_wrapper)
        graph.add_node("decide", self._decide_wrapper)
        graph.add_node("execute", self._execute_wrapper)
        graph.add_node("learn", self._learn_wrapper)
        graph.add_node("finalize", self._finalize_cycle)
        
        # Define the main flow
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "observe")
        graph.add_edge("observe", "recall")
        graph.add_edge("recall", "analyze")
        graph.add_edge("analyze", "decide")
        
        # Conditional routing from decide node
        graph.add_conditional_edges(
            "decide",
            self._route_decision,
            {
                "execute": "execute",
                "skip": "finalize",
                "emergency": END  # Emergency exit
            }
        )
        
        graph.add_edge("execute", "learn")
        graph.add_edge("learn", "finalize") 
        graph.add_edge("finalize", END)
        
        return graph

    async def _initialize_cycle(self, state: BrainState) -> BrainState:
        """Initialize a new brain cycle with metadata and safety checks."""
        
        cycle_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Check for emergency stop
        if self.emergency_stop_active:
            resume_time = state.get('resume_time')
            if resume_time and timestamp < resume_time:
                self.logger.warning("Brain in emergency stop mode, skipping cycle")
                return {
                    **state,
                    'emergency_stop_active': True,
                    'cycle_id': cycle_id,
                    'timestamp': timestamp
                }
        
        # Reset emergency stop if time has passed
        if self.emergency_stop_active and timestamp >= state.get('resume_time', timestamp):
            self.emergency_stop_active = False
            self.logger.info("Emergency stop lifted, resuming normal operation")
        
        # Initialize cycle state
        new_state = {
            'cycle_id': cycle_id,
            'timestamp': timestamp,
            'cycle_count': state.get('cycle_count', 0) + 1,
            'node_execution_times': {},
            'debug_logs': [f"Starting cycle {cycle_id}"],
            'warnings': [],
            'errors': [],
            'emergency_stop_active': self.emergency_stop_active
        }
        
        self.logger.info(f"Initialized brain cycle {cycle_id}")
        return {**state, **new_state}

    async def _observe_wrapper(self, state: BrainState) -> BrainState:
        """Wrapper for observation node with timing and error handling."""
        
        start_time = time.time()
        
        try:
            result = await self.observer.observe(state)
            execution_time = time.time() - start_time
            
            # Update execution times
            node_times = state.get('node_execution_times', {})
            node_times['observe'] = execution_time
            
            return {
                **result,
                'node_execution_times': node_times,
                'debug_logs': state.get('debug_logs', []) + [
                    f"Observation completed in {execution_time:.2f}s"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in observation node: {e}")
            return await self._handle_node_error("observe", e, state)

    async def _recall_wrapper(self, state: BrainState) -> BrainState:
        """Wrapper for memory recall node."""
        
        start_time = time.time()
        
        try:
            result = await self.recall.recall_memories(state)
            execution_time = time.time() - start_time
            
            node_times = state.get('node_execution_times', {})
            node_times['recall'] = execution_time
            
            return {
                **result,
                'node_execution_times': node_times,
                'debug_logs': state.get('debug_logs', []) + [
                    f"Memory recall completed in {execution_time:.2f}s"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in recall node: {e}")
            return await self._handle_node_error("recall", e, state)

    async def _analyze_wrapper(self, state: BrainState) -> BrainState:
        """Wrapper for opportunity analysis node."""
        
        start_time = time.time()
        
        try:
            result = await self.analyzer.analyze_opportunities(state)
            execution_time = time.time() - start_time
            
            node_times = state.get('node_execution_times', {})
            node_times['analyze'] = execution_time
            
            return {
                **result,
                'node_execution_times': node_times,
                'debug_logs': state.get('debug_logs', []) + [
                    f"Analysis completed in {execution_time:.2f}s"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in analysis node: {e}")
            return await self._handle_node_error("analyze", e, state)

    async def _decide_wrapper(self, state: BrainState) -> BrainState:
        """Wrapper for decision making node."""
        
        start_time = time.time()
        
        try:
            result = await self.decision_maker.make_decision(state)
            execution_time = time.time() - start_time
            
            node_times = state.get('node_execution_times', {})
            node_times['decide'] = execution_time
            
            return {
                **result,
                'node_execution_times': node_times,
                'debug_logs': state.get('debug_logs', []) + [
                    f"Decision completed in {execution_time:.2f}s"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in decision node: {e}")
            return await self._handle_node_error("decide", e, state)

    async def _execute_wrapper(self, state: BrainState) -> BrainState:
        """Wrapper for action execution node."""
        
        start_time = time.time()
        
        try:
            result = await self.executor.execute_action(state)
            execution_time = time.time() - start_time
            
            node_times = state.get('node_execution_times', {})
            node_times['execute'] = execution_time
            
            return {
                **result,
                'node_execution_times': node_times,
                'debug_logs': state.get('debug_logs', []) + [
                    f"Execution completed in {execution_time:.2f}s"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in execution node: {e}")
            return await self._handle_node_error("execute", e, state)

    async def _learn_wrapper(self, state: BrainState) -> BrainState:
        """Wrapper for learning node."""
        
        start_time = time.time()
        
        try:
            result = await self.learner.learn_from_result(state)
            execution_time = time.time() - start_time
            
            node_times = state.get('node_execution_times', {})
            node_times['learn'] = execution_time
            
            return {
                **result,
                'node_execution_times': node_times,
                'debug_logs': state.get('debug_logs', []) + [
                    f"Learning completed in {execution_time:.2f}s"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in learning node: {e}")
            return await self._handle_node_error("learn", e, state)

    async def _finalize_cycle(self, state: BrainState) -> BrainState:
        """Finalize the brain cycle with cleanup and metrics."""
        
        total_time = sum(state.get('node_execution_times', {}).values())
        
        # Update performance metrics
        metrics = state.get('performance_metrics', {})
        metrics.update({
            'cycle_time': total_time,
            'last_cycle_timestamp': state['timestamp'].isoformat(),
            'cycles_completed': state.get('cycle_count', 0)
        })
        
        # Log cycle completion
        self.logger.info(
            f"Brain cycle {state['cycle_id']} completed in {total_time:.2f}s"
        )
        
        return {
            **state,
            'performance_metrics': metrics,
            'debug_logs': state.get('debug_logs', []) + [
                f"Cycle {state['cycle_id']} completed successfully"
            ]
        }

    def _route_decision(self, state: BrainState) -> Literal["execute", "skip", "emergency"]:
        """
        Route decision based on selected action and risk assessment.
        
        Uses modern LangGraph routing pattern with conditional edges.
        """
        
        # Check for emergency conditions
        if state.get('emergency_stop_active', False):
            return "emergency"
        
        # Check if we have a valid action selected
        selected_action = state.get('selected_action')
        if not selected_action:
            self.logger.info("No action selected, skipping execution")
            return "skip"
        
        # Check confidence and risk thresholds
        confidence = state.get('confidence_score', 0)
        risk_assessment = state.get('risk_assessment', {})
        
        if confidence < self.config.confidence_threshold:
            self.logger.info(f"Confidence too low: {confidence}")
            return "skip"
        
        if not risk_assessment.get('acceptable', False):
            self.logger.info("Risk assessment failed")
            return "skip"
        
        # All checks passed, proceed with execution
        self.logger.info("All checks passed, proceeding with execution")
        return "execute"

    async def _handle_node_error(
        self, 
        node_name: str, 
        error: Exception, 
        state: BrainState
    ) -> BrainState:
        """Handle errors that occur in individual nodes."""
        
        error_info = {
            'node': node_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now()
        }
        
        # Use error handler for recovery
        recovery_result = await self.error_handler.handle_error(error, state)
        
        # Update state with error information
        errors = state.get('errors', [])
        errors.append(error_info)
        
        return {
            **state,
            'errors': errors,
            'error_count': state.get('error_count', 0) + 1,
            'recovery_action': recovery_result
        }

    async def run_single_cycle(self, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a single brain cycle.
        
        Args:
            initial_state: Optional initial state to start with
            
        Returns:
            Final state after cycle completion
        """
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        # Prepare initial state
        if initial_state is None:
            initial_state = {}
        
        try:
            # Run the graph
            result = await self.compiled_graph.ainvoke(initial_state, config)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in brain cycle: {e}")
            
            # Handle critical errors
            if self._is_critical_error(e):
                await self._activate_emergency_stop()
            
            raise

    async def run_continuous(self, interval: Optional[int] = None) -> None:
        """
        Run the brain continuously in cycles.
        
        Args:
            interval: Sleep interval between cycles (uses config default if None)
        """
        
        if interval is None:
            interval = self.config.observation_interval
        
        self.is_running = True
        self.logger.info(f"Starting continuous brain operation (interval: {interval}s)")
        
        try:
            while self.is_running:
                if not self.emergency_stop_active:
                    try:
                        await self.run_single_cycle()
                        self.cycle_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error in continuous cycle: {e}")
                        
                        # Sleep longer after errors
                        await asyncio.sleep(interval * 2)
                        continue
                
                # Wait before next cycle
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Brain operation stopped by user")
        
        finally:
            self.is_running = False

    async def stop(self):
        """Stop continuous brain operation gracefully."""
        
        self.logger.info("Stopping brain operation...")
        self.is_running = False

    async def _activate_emergency_stop(self):
        """Activate emergency stop procedures."""
        
        self.emergency_stop_active = True
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        
        # Stop continuous operation
        self.is_running = False
        
        # Set resume time (1 hour from now)
        resume_time = datetime.now() + timedelta(hours=1)
        
        # Cancel any pending transactions
        try:
            if hasattr(self.executor, 'cancel_pending_transactions'):
                await self.executor.cancel_pending_transactions()
        except Exception as e:
            self.logger.error(f"Error canceling transactions: {e}")

    def _is_critical_error(self, error: Exception) -> bool:
        """Determine if an error is critical enough to trigger emergency stop."""
        
        error_msg = str(error).lower()
        critical_patterns = [
            'insufficient funds',
            'transaction failed',
            'network error',
            'timeout',
            'connection lost'
        ]
        
        return any(pattern in error_msg for pattern in critical_patterns)

    async def get_status(self) -> Dict[str, Any]:
        """Get current brain status and metrics."""
        
        return {
            'is_running': self.is_running,
            'emergency_stop_active': self.emergency_stop_active,
            'cycle_count': self.cycle_count,
            'config': {
                'confidence_threshold': self.config.confidence_threshold,
                'risk_threshold': self.config.risk_threshold,
                'observation_interval': self.config.observation_interval
            },
            'components': {
                'cdp_manager_connected': self.cdp_manager is not None,
                'memory_system_connected': self.memory_system is not None
            }
        }

    def __repr__(self) -> str:
        return f"AerodromeBrain(running={self.is_running}, cycles={self.cycle_count})"


# Factory function for easy brain creation
async def create_brain(
    config: Optional[BrainConfig] = None,
    cdp_manager=None,
    memory_system=None
) -> AerodromeBrain:
    """
    Factory function to create and initialize an AerodromeBrain instance.
    
    Args:
        config: Brain configuration
        cdp_manager: CDP SDK manager
        memory_system: Memory system instance
        
    Returns:
        Initialized AerodromeBrain instance
    """
    
    brain = AerodromeBrain(
        config=config,
        cdp_manager=cdp_manager,
        memory_system=memory_system
    )
    
    return brain