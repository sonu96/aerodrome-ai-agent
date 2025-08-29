"""
Brain state definitions for LangGraph implementation.

This module defines the state schema and configuration for the Aerodrome AI agent's brain,
following LangGraph best practices for state management and type safety.
"""

from typing import Dict, List, Any, Optional, Annotated
from typing_extensions import TypedDict
from datetime import datetime
from dataclasses import dataclass
import operator


@dataclass
class BrainConfig:
    """Configuration for the Aerodrome Brain"""
    
    # Decision thresholds
    confidence_threshold: float = 0.7
    risk_threshold: float = 0.3
    min_opportunity_score: float = 0.6
    
    # Timing parameters
    observation_interval: int = 60  # seconds
    execution_timeout: int = 30  # seconds
    max_cycle_time: int = 300  # 5 minutes max per cycle
    
    # Memory configuration
    max_memories_per_query: int = 10
    pattern_extraction_threshold: int = 5
    memory_retention_days: int = 30
    
    # Safety parameters
    max_position_size: float = 0.2  # 20% of portfolio
    max_slippage: float = 0.02  # 2%
    emergency_stop_loss: float = 0.1  # 10%
    max_gas_price: float = 100  # gwei
    
    # Performance tuning
    max_concurrent_analysis: int = 5
    cache_ttl_seconds: int = 60
    simulation_required: bool = True
    
    # Network configuration
    network: str = "base"
    rpc_timeout: int = 10
    max_retries: int = 3


class BrainState(TypedDict, total=False):
    """
    Complete state representation for brain operations.
    Using TypedDict with total=False allows optional fields while maintaining type safety.
    """
    
    # === Market Context ===
    timestamp: datetime
    cycle_id: str
    market_data: Dict[str, Any]
    gas_price: float
    network_congestion: float
    base_fee_trend: str  # "increasing", "stable", "decreasing"
    
    # === Portfolio State ===
    wallet_address: str
    wallet_balance: Dict[str, float]  # token_address -> balance
    active_positions: Annotated[List[Dict[str, Any]], operator.add]
    pending_transactions: List[str]
    total_value_locked: float
    portfolio_performance: Dict[str, float]
    
    # === Analysis Results ===
    opportunities: Annotated[List[Dict[str, Any]], operator.add]
    risk_scores: Dict[str, float]
    expected_returns: Dict[str, float]
    market_sentiment: Dict[str, Any]
    pool_analytics: Dict[str, Dict[str, Any]]
    
    # === Decision State ===
    selected_action: Optional[Dict[str, Any]]
    confidence_score: float
    risk_assessment: Dict[str, Any]
    execution_plan: Optional[Dict[str, Any]]
    decision_reasoning: List[str]
    
    # === Execution State ===
    transaction_params: Optional[Dict[str, Any]]
    simulation_result: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    gas_estimate: Optional[float]
    
    # === Memory Context ===
    relevant_memories: Annotated[List[Dict[str, Any]], operator.add]
    learned_patterns: Annotated[List[Dict[str, Any]], operator.add]
    failure_patterns: Annotated[List[Dict[str, Any]], operator.add]
    recent_experiences: Annotated[List[Dict[str, Any]], operator.add]
    
    # === Meta State ===
    cycle_count: int
    last_action_time: Optional[datetime]
    performance_metrics: Dict[str, float]
    error_count: int
    emergency_stop_active: bool
    resume_time: Optional[datetime]
    
    # === Debug and Logging ===
    node_execution_times: Dict[str, float]
    debug_logs: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]
    errors: Annotated[List[Dict[str, Any]], operator.add]


class OpportunityScore(TypedDict):
    """Structure for opportunity scoring results"""
    
    # Individual factor scores (0-1)
    yield_score: float
    volume_score: float
    tvl_score: float
    historical_score: float
    pattern_score: float
    timing_score: float
    
    # Weighted total score (0-1)
    total_score: float
    
    # Confidence in the score
    confidence: float
    
    # Factors used in calculation
    factors_considered: List[str]


class RiskAssessment(TypedDict):
    """Structure for risk assessment results"""
    
    # Risk category scores (0-1, where 1 is highest risk)
    position_risk: float
    market_risk: float
    liquidity_risk: float
    contract_risk: float
    timing_risk: float
    impermanent_loss_risk: float
    
    # Overall risk score (0-1)
    total_risk: float
    
    # Risk acceptability
    acceptable: bool
    recommendation: str  # "proceed", "caution", "abort"
    
    # Risk mitigation suggestions
    mitigations: List[str]
    
    # Risk factors identified
    risk_factors: List[str]


class ActionPlan(TypedDict):
    """Structure for action execution plans"""
    
    action_type: str  # "SWAP", "ADD_LIQUIDITY", "REMOVE_LIQUIDITY", "VOTE", "CLAIM"
    pool_address: str
    token_addresses: List[str]
    amounts: Dict[str, float]
    expected_outcome: Dict[str, Any]
    
    # Execution parameters
    slippage_tolerance: float
    deadline_minutes: int
    gas_strategy: str  # "fast", "standard", "slow"
    
    # Risk parameters
    max_acceptable_loss: float
    stop_loss_threshold: Optional[float]
    
    # Monitoring requirements
    monitor_duration: int  # seconds
    success_criteria: Dict[str, Any]


class ExecutionResult(TypedDict):
    """Structure for execution results"""
    
    success: bool
    transaction_hash: Optional[str]
    gas_used: Optional[int]
    gas_price_paid: Optional[float]
    execution_time: float
    
    # Financial outcome
    tokens_received: Dict[str, float]
    tokens_spent: Dict[str, float]
    net_value_change: float
    slippage_experienced: float
    
    # Error information (if failed)
    error_type: Optional[str]
    error_message: Optional[str]
    recovery_attempted: bool
    
    # Post-execution state
    new_balances: Dict[str, float]
    new_positions: List[Dict[str, Any]]


class MemoryEntry(TypedDict):
    """Structure for memory entries"""
    
    timestamp: datetime
    memory_type: str  # "experience", "pattern", "failure", "success"
    
    # Context
    market_conditions: Dict[str, Any]
    action_taken: Dict[str, Any]
    outcome: Dict[str, Any]
    
    # Analysis
    lessons_learned: List[str]
    pattern_features: Dict[str, Any]
    success_factors: List[str]
    failure_reasons: List[str]
    
    # Metadata
    confidence_level: float
    relevance_score: float
    usage_count: int
    last_accessed: datetime


# State update helper functions for reducers
def add_opportunities(existing: Optional[List] = None, new: Optional[List] = None) -> List:
    """Add new opportunities to existing list, removing duplicates by pool address"""
    if existing is None:
        existing = []
    if new is None:
        return existing
    
    # Create a map by pool address to avoid duplicates
    opp_map = {opp['pool']['address']: opp for opp in existing}
    
    # Add new opportunities
    for opp in new:
        pool_addr = opp['pool']['address']
        opp_map[pool_addr] = opp
    
    # Return sorted by score
    return sorted(opp_map.values(), key=lambda x: x['score']['total_score'], reverse=True)


def add_memories(existing: Optional[List] = None, new: Optional[List] = None) -> List:
    """Add new memories to existing list with deduplication and size limits"""
    if existing is None:
        existing = []
    if new is None:
        return existing
    
    # Combine and deduplicate by timestamp and type
    combined = existing + new
    
    # Remove duplicates while preserving order
    seen = set()
    unique_memories = []
    
    for memory in combined:
        key = (memory['timestamp'].isoformat(), memory['memory_type'])
        if key not in seen:
            seen.add(key)
            unique_memories.append(memory)
    
    # Sort by relevance and recency
    unique_memories.sort(
        key=lambda x: (x.get('relevance_score', 0.5), x['timestamp']), 
        reverse=True
    )
    
    # Limit to reasonable size (keep most relevant)
    return unique_memories[:50]


def add_debug_logs(existing: Optional[List] = None, new: Optional[List] = None) -> List:
    """Add debug logs with timestamp and size limits"""
    if existing is None:
        existing = []
    if new is None:
        return existing
    
    combined = existing + new
    
    # Keep only recent logs (last 100)
    return combined[-100:]


# Export all types for easy importing
__all__ = [
    'BrainConfig',
    'BrainState', 
    'OpportunityScore',
    'RiskAssessment',
    'ActionPlan',
    'ExecutionResult',
    'MemoryEntry',
    'add_opportunities',
    'add_memories',
    'add_debug_logs'
]