"""
Brain State Definition

Defines the complete state representation for brain operations using TypedDict
for type safety and clear data structure definition.
"""

from typing import TypedDict, Dict, List, Any, Optional
from datetime import datetime


class BrainState(TypedDict):
    """Complete state representation for brain operations"""
    
    # Market Context
    timestamp: datetime
    cycle_count: int
    market_data: Dict[str, Any]
    gas_price: float
    network_congestion: float
    
    # Portfolio State  
    wallet_balance: Dict[str, float]
    active_positions: List[Dict[str, Any]]
    pending_transactions: List[str]
    total_value_locked: float
    portfolio_value: float
    
    # Analysis Results
    opportunities: List[Dict[str, Any]]
    risk_scores: Dict[str, float]
    expected_returns: Dict[str, float]
    market_trends: Dict[str, Any]
    
    # Decision State
    selected_action: Optional[Dict[str, Any]]
    confidence_score: float
    risk_assessment: Dict[str, Any]
    execution_plan: Optional[Dict[str, Any]]
    decision_rationale: str
    
    # Execution State
    transaction_params: Optional[Dict[str, Any]]
    simulation_result: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    execution_status: str
    
    # Memory Context
    relevant_memories: List[Dict[str, Any]]
    learned_patterns: List[Dict[str, Any]]
    failure_patterns: List[Dict[str, Any]]
    pattern_confidence: float
    
    # Performance Metrics
    performance_metrics: Dict[str, float]
    success_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Meta State
    last_action_time: datetime
    emergency_stop_active: bool
    safe_mode: bool
    errors: List[Dict[str, Any]]
    warnings: List[str]


class PoolData(TypedDict):
    """Pool data structure"""
    address: str
    token0: str
    token1: str
    stable: bool
    reserves: List[int]
    total_supply: int
    apr: float
    emissions_apr: float
    volume_24h: float
    volume_7d_avg: float
    tvl: float
    tvl_change_24h: float
    fees_24h: float
    created_at: datetime


class OpportunityScore(TypedDict):
    """Opportunity scoring breakdown"""
    total_score: float
    apr_score: float
    volume_score: float
    tvl_score: float
    historical_score: float
    pattern_score: float
    risk_adjusted_score: float


class RiskAssessment(TypedDict):
    """Risk assessment breakdown"""
    total_risk: float
    position_risk: float
    market_risk: float
    liquidity_risk: float
    contract_risk: float
    impermanent_loss_risk: float
    acceptable: bool
    recommendation: str
    mitigations: List[str]


class ExecutionPlan(TypedDict):
    """Execution plan details"""
    action_type: str
    pool: PoolData
    amount: float
    amount_usd: float
    expected_outcome: Dict[str, Any]
    gas_estimate: int
    slippage_tolerance: float
    deadline: int
    steps: List[Dict[str, Any]]


def create_initial_state() -> BrainState:
    """Create initial brain state with default values"""
    return BrainState(
        # Market Context
        timestamp=datetime.now(),
        cycle_count=0,
        market_data={},
        gas_price=0.0,
        network_congestion=0.0,
        
        # Portfolio State
        wallet_balance={},
        active_positions=[],
        pending_transactions=[],
        total_value_locked=0.0,
        portfolio_value=0.0,
        
        # Analysis Results
        opportunities=[],
        risk_scores={},
        expected_returns={},
        market_trends={},
        
        # Decision State
        selected_action=None,
        confidence_score=0.0,
        risk_assessment={},
        execution_plan=None,
        decision_rationale="",
        
        # Execution State
        transaction_params=None,
        simulation_result=None,
        execution_result=None,
        execution_status="idle",
        
        # Memory Context
        relevant_memories=[],
        learned_patterns=[],
        failure_patterns=[],
        pattern_confidence=0.0,
        
        # Performance Metrics
        performance_metrics={},
        success_rate=0.0,
        avg_return=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        
        # Meta State
        last_action_time=datetime.now(),
        emergency_stop_active=False,
        safe_mode=False,
        errors=[],
        warnings=[]
    )