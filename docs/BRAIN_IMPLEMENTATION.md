# Brain Implementation Guide

## Table of Contents
1. [Brain Core Architecture](#brain-core-architecture)
2. [LangGraph State Machine](#langgraph-state-machine)
3. [Node Implementations](#node-implementations)
4. [Decision Algorithms](#decision-algorithms)
5. [Execution Patterns](#execution-patterns)
6. [Error Handling](#error-handling)

## Brain Core Architecture

### Overview
The brain is implemented as a directed acyclic graph (DAG) using LangGraph, where each node represents a cognitive function and edges define the flow of decision-making.

### Core Brain Class

```python
from typing import TypedDict, Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from dataclasses import dataclass
import asyncio
from datetime import datetime
import numpy as np

@dataclass
class BrainConfig:
    """Configuration for the Aerodrome Brain"""
    
    # Decision thresholds
    confidence_threshold: float = 0.7
    risk_threshold: float = 0.3
    min_opportunity_score: float = 0.6
    
    # Timing
    observation_interval: int = 60  # seconds
    execution_timeout: int = 30  # seconds
    
    # Memory
    max_memories_per_query: int = 10
    pattern_extraction_threshold: int = 5
    
    # Safety
    max_position_size: float = 0.2  # 20% of portfolio
    max_slippage: float = 0.02  # 2%
    emergency_stop_loss: float = 0.1  # 10%

class BrainState(TypedDict):
    """Complete state representation for brain operations"""
    
    # Market Context
    timestamp: datetime
    market_data: Dict[str, Any]
    gas_price: float
    network_congestion: float
    
    # Portfolio State
    wallet_balance: Dict[str, float]
    active_positions: List[Dict[str, Any]]
    pending_transactions: List[str]
    total_value_locked: float
    
    # Analysis Results
    opportunities: List[Dict[str, Any]]
    risk_scores: Dict[str, float]
    expected_returns: Dict[str, float]
    
    # Decision State
    selected_action: Optional[Dict[str, Any]]
    confidence_score: float
    risk_assessment: Dict[str, Any]
    execution_plan: Optional[Dict[str, Any]]
    
    # Execution State
    transaction_params: Optional[Dict[str, Any]]
    simulation_result: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    
    # Memory Context
    relevant_memories: List[Dict[str, Any]]
    learned_patterns: List[Dict[str, Any]]
    failure_patterns: List[Dict[str, Any]]
    
    # Meta State
    cycle_count: int
    last_action_time: datetime
    performance_metrics: Dict[str, float]
```

## LangGraph State Machine

### Graph Construction

```python
class AerodromeBrain:
    def __init__(self, config: BrainConfig = None):
        self.config = config or BrainConfig()
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()
        
    def _build_graph(self) -> StateGraph:
        """Construct the brain's cognitive graph"""
        
        graph = StateGraph(BrainState)
        
        # Add all cognitive nodes
        graph.add_node("initialize", self.initialize_state)
        graph.add_node("observe", self.observe_market)
        graph.add_node("recall", self.recall_memories)
        graph.add_node("analyze", self.analyze_opportunities)
        graph.add_node("strategize", self.form_strategy)
        graph.add_node("assess_risk", self.assess_risk)
        graph.add_node("decide", self.make_decision)
        graph.add_node("simulate", self.simulate_execution)
        graph.add_node("execute", self.execute_action)
        graph.add_node("monitor", self.monitor_execution)
        graph.add_node("learn", self.learn_from_result)
        graph.add_node("update_memory", self.update_memory)
        graph.add_node("prune", self.prune_memories)
        
        # Define edges
        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "observe")
        graph.add_edge("observe", "recall")
        graph.add_edge("recall", "analyze")
        graph.add_edge("analyze", "strategize")
        graph.add_edge("strategize", "assess_risk")
        graph.add_edge("assess_risk", "decide")
        
        # Conditional routing from decide node
        graph.add_conditional_edges(
            "decide",
            self.route_decision,
            {
                "simulate": "simulate",
                "skip": "update_memory",
                "emergency": "monitor"
            }
        )
        
        # Conditional routing from simulate node
        graph.add_conditional_edges(
            "simulate",
            self.route_simulation,
            {
                "execute": "execute",
                "reject": "update_memory",
                "retry": "strategize"
            }
        )
        
        graph.add_edge("execute", "monitor")
        graph.add_edge("monitor", "learn")
        graph.add_edge("learn", "update_memory")
        graph.add_edge("update_memory", "prune")
        graph.add_edge("prune", END)
        
        return graph
    
    def route_decision(self, state: BrainState) -> str:
        """Routing logic for decision node"""
        
        if state.get('selected_action') is None:
            return "skip"
        
        if state['risk_assessment'].get('emergency', False):
            return "emergency"
        
        if state['confidence_score'] >= self.config.confidence_threshold:
            return "simulate"
        
        return "skip"
    
    def route_simulation(self, state: BrainState) -> str:
        """Routing logic for simulation results"""
        
        sim_result = state.get('simulation_result', {})
        
        if sim_result.get('success', False):
            if sim_result.get('profitable', False):
                return "execute"
            else:
                return "retry"  # Try different strategy
        
        return "reject"  # Simulation failed
```

## Node Implementations

### Observation Node

```python
async def observe_market(self, state: BrainState) -> BrainState:
    """Observe current market conditions and portfolio state"""
    
    # Parallel data collection
    tasks = [
        self._get_pool_data(),
        self._get_wallet_balances(),
        self._get_active_positions(),
        self._get_gas_price(),
        self._get_pending_transactions()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    pools, balances, positions, gas, pending = results
    
    # Calculate derived metrics
    tvl = sum(pos['value_usd'] for pos in positions if not isinstance(pos, Exception))
    
    # Update state
    state.update({
        'timestamp': datetime.now(),
        'market_data': self._process_pool_data(pools),
        'wallet_balance': balances,
        'active_positions': positions,
        'gas_price': gas,
        'pending_transactions': pending,
        'total_value_locked': tvl,
        'network_congestion': self._calculate_congestion(gas)
    })
    
    return state

async def _get_pool_data(self) -> List[Dict]:
    """Fetch all Aerodrome pool data via CDP SDK"""
    
    pools = []
    
    # Get top pools by TVL
    top_pools = await self.cdp.readContract({
        'contractAddress': FACTORY_ADDRESS,
        'method': 'allPools',
        'abi': FACTORY_ABI
    })
    
    # Fetch detailed data for each pool
    for pool_address in top_pools[:20]:  # Limit to top 20
        pool_data = await self._get_single_pool_data(pool_address)
        pools.append(pool_data)
    
    return pools

async def _get_single_pool_data(self, pool_address: str) -> Dict:
    """Get comprehensive data for a single pool"""
    
    # Parallel contract reads
    tasks = [
        self.cdp.readContract({
            'contractAddress': pool_address,
            'method': 'getReserves',
            'abi': POOL_ABI
        }),
        self.cdp.readContract({
            'contractAddress': pool_address,
            'method': 'stable',
            'abi': POOL_ABI
        }),
        self.cdp.readContract({
            'contractAddress': pool_address,
            'method': 'totalSupply',
            'abi': POOL_ABI
        }),
        self._get_pool_apr(pool_address)
    ]
    
    reserves, is_stable, total_supply, apr = await asyncio.gather(*tasks)
    
    return {
        'address': pool_address,
        'reserves': reserves,
        'stable': is_stable,
        'total_supply': total_supply,
        'apr': apr,
        'tvl': self._calculate_tvl(reserves),
        'volume_24h': await self._get_24h_volume(pool_address)
    }
```

### Analysis Node

```python
async def analyze_opportunities(self, state: BrainState) -> BrainState:
    """Analyze market for profitable opportunities"""
    
    opportunities = []
    
    # Analyze each pool
    for pool in state['market_data']:
        opp_score = await self._score_opportunity(pool, state)
        
        if opp_score['total_score'] >= self.config.min_opportunity_score:
            opportunities.append({
                'pool': pool,
                'score': opp_score,
                'action_type': self._determine_action(pool, state),
                'expected_return': self._calculate_expected_return(pool, opp_score)
            })
    
    # Sort by score
    opportunities.sort(key=lambda x: x['score']['total_score'], reverse=True)
    
    # Update state
    state['opportunities'] = opportunities
    state['expected_returns'] = {
        opp['pool']['address']: opp['expected_return'] 
        for opp in opportunities
    }
    
    return state

async def _score_opportunity(self, pool: Dict, state: BrainState) -> Dict:
    """Multi-factor opportunity scoring"""
    
    scores = {}
    
    # APR Score (0-1)
    apr = pool.get('apr', 0)
    scores['apr_score'] = min(apr / 100, 1.0)  # Normalize to 100% APR
    
    # Volume Score (0-1)
    volume = pool.get('volume_24h', 0)
    avg_volume = np.mean([p['volume_24h'] for p in state['market_data']])
    scores['volume_score'] = min(volume / (avg_volume * 2), 1.0)
    
    # TVL Score (0-1) - Higher TVL = More stable
    tvl = pool.get('tvl', 0)
    scores['tvl_score'] = min(tvl / 1_000_000, 1.0)  # Normalize to $1M
    
    # Historical Performance Score
    memories = state['relevant_memories']
    hist_score = self._calculate_historical_score(pool, memories)
    scores['historical_score'] = hist_score
    
    # Pattern Match Score
    patterns = state['learned_patterns']
    pattern_score = self._calculate_pattern_match(pool, patterns)
    scores['pattern_score'] = pattern_score
    
    # Weighted total score
    weights = {
        'apr_score': 0.3,
        'volume_score': 0.2,
        'tvl_score': 0.15,
        'historical_score': 0.2,
        'pattern_score': 0.15
    }
    
    total_score = sum(scores[key] * weights[key] for key in weights)
    scores['total_score'] = total_score
    
    return scores
```

### Decision Node

```python
async def make_decision(self, state: BrainState) -> BrainState:
    """Make final decision based on analysis and risk assessment"""
    
    if not state['opportunities']:
        state['selected_action'] = None
        state['confidence_score'] = 0
        return state
    
    # Get best opportunity
    best_opp = state['opportunities'][0]
    
    # Calculate confidence
    confidence = self._calculate_confidence(best_opp, state)
    
    # Check risk constraints
    risk_check = self._check_risk_constraints(best_opp, state)
    
    if confidence >= self.config.confidence_threshold and risk_check['passed']:
        # Prepare action
        action = self._prepare_action(best_opp, state)
        
        state['selected_action'] = action
        state['confidence_score'] = confidence
        state['execution_plan'] = self._create_execution_plan(action, state)
        
        # Build transaction parameters
        state['transaction_params'] = self._build_transaction_params(action, state)
    else:
        state['selected_action'] = None
        state['confidence_score'] = confidence
        
        # Log why decision was rejected
        self.logger.info(f"Decision rejected: confidence={confidence}, risk_check={risk_check}")
    
    return state

def _calculate_confidence(self, opportunity: Dict, state: BrainState) -> float:
    """Calculate confidence score for decision"""
    
    base_confidence = opportunity['score']['total_score']
    
    # Adjust based on market conditions
    market_factor = 1.0
    
    # Check if market is volatile
    if self._is_market_volatile(state['market_data']):
        market_factor *= 0.8
    
    # Check gas prices
    if state['gas_price'] > 50:  # High gas
        market_factor *= 0.9
    
    # Check historical success rate
    success_rate = self._get_historical_success_rate(opportunity['pool'], state)
    history_factor = 0.5 + (success_rate * 0.5)  # 0.5 to 1.0
    
    # Check pattern strength
    pattern_strength = opportunity['score'].get('pattern_score', 0.5)
    pattern_factor = 0.7 + (pattern_strength * 0.3)  # 0.7 to 1.0
    
    confidence = base_confidence * market_factor * history_factor * pattern_factor
    
    return min(confidence, 1.0)
```

### Execution Node

```python
async def execute_action(self, state: BrainState) -> BrainState:
    """Execute the selected action via CDP SDK"""
    
    action = state['selected_action']
    params = state['transaction_params']
    
    try:
        # Pre-execution checks
        await self._pre_execution_checks(state)
        
        # Execute based on action type
        if action['type'] == 'SWAP':
            result = await self._execute_swap(params)
            
        elif action['type'] == 'ADD_LIQUIDITY':
            result = await self._execute_add_liquidity(params)
            
        elif action['type'] == 'REMOVE_LIQUIDITY':
            result = await self._execute_remove_liquidity(params)
            
        elif action['type'] == 'VOTE':
            result = await self._execute_vote(params)
            
        elif action['type'] == 'CLAIM':
            result = await self._execute_claim(params)
            
        else:
            raise ValueError(f"Unknown action type: {action['type']}")
        
        # Wait for confirmation
        confirmed = await self._wait_for_confirmation(result['tx_hash'])
        
        state['execution_result'] = {
            'success': True,
            'tx_hash': result['tx_hash'],
            'gas_used': confirmed['gas_used'],
            'timestamp': datetime.now(),
            'details': confirmed
        }
        
    except Exception as e:
        state['execution_result'] = {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(),
            'action': action
        }
        
        # Trigger emergency procedures if needed
        if self._is_critical_error(e):
            await self._emergency_stop(state)
    
    return state

async def _execute_swap(self, params: Dict) -> Dict:
    """Execute token swap via CDP SDK"""
    
    result = await self.cdp.wallet.invokeContract({
        'contractAddress': AERODROME_ROUTER,
        'method': 'swapExactTokensForTokens',
        'args': {
            'amountIn': params['amount_in'],
            'amountOutMin': params['amount_out_min'],
            'routes': params['routes'],
            'to': params['to'],
            'deadline': params['deadline']
        },
        'abi': ROUTER_ABI
    })
    
    return {
        'tx_hash': result.transaction_hash,
        'raw_result': result
    }

async def _execute_add_liquidity(self, params: Dict) -> Dict:
    """Add liquidity to pool via CDP SDK"""
    
    result = await self.cdp.wallet.invokeContract({
        'contractAddress': AERODROME_ROUTER,
        'method': 'addLiquidity',
        'args': {
            'tokenA': params['token_a'],
            'tokenB': params['token_b'],
            'stable': params['stable'],
            'amountADesired': params['amount_a'],
            'amountBDesired': params['amount_b'],
            'amountAMin': params['amount_a_min'],
            'amountBMin': params['amount_b_min'],
            'to': params['to'],
            'deadline': params['deadline']
        },
        'abi': ROUTER_ABI
    })
    
    return {
        'tx_hash': result.transaction_hash,
        'raw_result': result
    }
```

## Decision Algorithms

### Opportunity Scoring Algorithm

```python
def calculate_opportunity_score(
    pool_data: Dict,
    market_conditions: Dict,
    historical_data: List[Dict],
    patterns: List[Dict]
) -> float:
    """
    Multi-factor opportunity scoring with weighted components
    
    Score = Σ(weight_i * factor_i)
    
    Factors:
    - APR/APY potential
    - Volume trends
    - TVL stability
    - Historical performance
    - Pattern matching
    - Market timing
    """
    
    scores = {}
    
    # 1. Yield Score (30% weight)
    apr = pool_data.get('apr', 0)
    emissions = pool_data.get('emissions_apr', 0)
    total_yield = apr + emissions
    
    # Sigmoid normalization for yields
    scores['yield'] = 2 / (1 + np.exp(-total_yield/20)) - 1
    
    # 2. Volume Trend Score (20% weight)
    volume_24h = pool_data.get('volume_24h', 0)
    volume_7d_avg = pool_data.get('volume_7d_avg', 0)
    
    if volume_7d_avg > 0:
        volume_trend = volume_24h / volume_7d_avg
        scores['volume'] = min(volume_trend / 2, 1.0)  # Cap at 2x average
    else:
        scores['volume'] = 0.5
    
    # 3. TVL Stability Score (15% weight)
    tvl = pool_data.get('tvl', 0)
    tvl_change_24h = pool_data.get('tvl_change_24h', 0)
    
    # Prefer stable or growing TVL
    if tvl > 100_000:  # Min $100k TVL
        stability = 1 - abs(tvl_change_24h) / 0.2  # Penalize >20% changes
        scores['stability'] = max(stability, 0)
    else:
        scores['stability'] = 0
    
    # 4. Historical Success Score (20% weight)
    success_rate = calculate_historical_success(pool_data['address'], historical_data)
    scores['historical'] = success_rate
    
    # 5. Pattern Match Score (15% weight)
    pattern_match = find_best_pattern_match(pool_data, patterns)
    scores['pattern'] = pattern_match
    
    # Weighted combination
    weights = {
        'yield': 0.30,
        'volume': 0.20,
        'stability': 0.15,
        'historical': 0.20,
        'pattern': 0.15
    }
    
    final_score = sum(scores[k] * weights[k] for k in weights)
    
    return final_score
```

### Risk Assessment Algorithm

```python
def assess_risk(
    action: Dict,
    portfolio: Dict,
    market_conditions: Dict
) -> Dict[str, Any]:
    """
    Comprehensive risk assessment for proposed action
    
    Risk Categories:
    - Position Risk: Size relative to portfolio
    - Market Risk: Volatility and liquidity
    - Smart Contract Risk: Protocol and pool specific
    - Timing Risk: Gas prices and congestion
    - Impermanent Loss Risk: For LP positions
    """
    
    risk_scores = {}
    
    # 1. Position Risk
    position_size = action['amount_usd']
    portfolio_value = portfolio['total_value']
    
    position_ratio = position_size / portfolio_value
    if position_ratio > 0.2:  # >20% of portfolio
        risk_scores['position'] = 0.8 + (position_ratio - 0.2)
    else:
        risk_scores['position'] = position_ratio * 4  # Linear up to 20%
    
    # 2. Market Risk
    pool = action['pool']
    volatility = calculate_pool_volatility(pool)
    
    risk_scores['market'] = volatility / 100  # Normalize 100% volatility to 1.0
    
    # 3. Liquidity Risk
    tvl = pool['tvl']
    trade_size = action['amount_usd']
    
    if tvl > 0:
        liquidity_impact = trade_size / tvl
        risk_scores['liquidity'] = min(liquidity_impact * 10, 1.0)
    else:
        risk_scores['liquidity'] = 1.0
    
    # 4. Smart Contract Risk
    pool_age = (datetime.now() - pool['created_at']).days
    
    if pool_age < 7:
        risk_scores['contract'] = 0.8
    elif pool_age < 30:
        risk_scores['contract'] = 0.5
    else:
        risk_scores['contract'] = 0.2
    
    # 5. Impermanent Loss Risk (for LP positions)
    if action['type'] == 'ADD_LIQUIDITY':
        il_risk = calculate_il_risk(pool)
        risk_scores['impermanent_loss'] = il_risk
    else:
        risk_scores['impermanent_loss'] = 0
    
    # Calculate aggregate risk
    total_risk = np.mean(list(risk_scores.values()))
    
    # Determine if acceptable
    acceptable = total_risk < 0.3  # 30% risk threshold
    
    return {
        'scores': risk_scores,
        'total_risk': total_risk,
        'acceptable': acceptable,
        'recommendation': 'proceed' if acceptable else 'abort',
        'mitigations': suggest_risk_mitigations(risk_scores)
    }
```

## Execution Patterns

### Transaction Building Pattern

```python
def build_transaction_params(action: Dict, state: BrainState) -> Dict:
    """
    Build optimized transaction parameters
    
    Considerations:
    - Gas optimization (EIP-1559)
    - Slippage protection
    - MEV protection
    - Deadline calculation
    """
    
    base_params = {
        'from': state['wallet_address'],
        'deadline': int(time.time()) + 900,  # 15 minutes
    }
    
    if action['type'] == 'SWAP':
        params = build_swap_params(action, state)
    elif action['type'] == 'ADD_LIQUIDITY':
        params = build_liquidity_params(action, state)
    else:
        params = {}
    
    # Add gas optimization
    gas_params = optimize_gas_params(state['gas_price'])
    params.update(gas_params)
    
    # Add MEV protection
    if action['amount_usd'] > 10000:  # Large trades
        params['private_pool'] = True  # Use private mempool if available
    
    return {**base_params, **params}

def optimize_gas_params(current_gas_price: float) -> Dict:
    """EIP-1559 gas optimization"""
    
    # Analyze recent blocks for base fee trend
    base_fee_trend = analyze_base_fee_trend()
    
    if base_fee_trend == 'increasing':
        # Be aggressive to get included quickly
        return {
            'maxFeePerGas': current_gas_price * 1.2,
            'maxPriorityFeePerGas': 2
        }
    else:
        # Can be more conservative
        return {
            'maxFeePerGas': current_gas_price * 1.05,
            'maxPriorityFeePerGas': 1
        }
```

## Error Handling

### Error Recovery Strategies

```python
class ErrorHandler:
    """Comprehensive error handling and recovery"""
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_strategies = {
            'INSUFFICIENT_BALANCE': self.handle_insufficient_balance,
            'SLIPPAGE_EXCEEDED': self.handle_slippage,
            'GAS_TOO_HIGH': self.handle_high_gas,
            'TRANSACTION_FAILED': self.handle_failed_transaction,
            'NETWORK_ERROR': self.handle_network_error
        }
    
    async def handle_error(self, error: Exception, state: BrainState) -> Dict:
        """Main error handling dispatcher"""
        
        error_type = self.classify_error(error)
        
        # Track error frequency
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Get recovery strategy
        strategy = self.recovery_strategies.get(
            error_type, 
            self.handle_unknown_error
        )
        
        # Execute recovery
        recovery_result = await strategy(error, state)
        
        # Log to memory for learning
        await self.log_error_to_memory(error, recovery_result)
        
        return recovery_result
    
    def classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate handling"""
        
        error_msg = str(error).lower()
        
        if 'insufficient' in error_msg or 'balance' in error_msg:
            return 'INSUFFICIENT_BALANCE'
        elif 'slippage' in error_msg:
            return 'SLIPPAGE_EXCEEDED'
        elif 'gas' in error_msg and 'high' in error_msg:
            return 'GAS_TOO_HIGH'
        elif 'revert' in error_msg or 'failed' in error_msg:
            return 'TRANSACTION_FAILED'
        elif 'timeout' in error_msg or 'network' in error_msg:
            return 'NETWORK_ERROR'
        else:
            return 'UNKNOWN'
    
    async def handle_slippage(self, error: Exception, state: BrainState) -> Dict:
        """Handle slippage exceeded errors"""
        
        current_slippage = state['transaction_params'].get('slippage', 0.01)
        
        if current_slippage < 0.05:  # Less than 5%
            # Increase slippage and retry
            new_slippage = min(current_slippage * 2, 0.05)
            state['transaction_params']['slippage'] = new_slippage
            
            return {
                'action': 'RETRY',
                'modifications': {'slippage': new_slippage}
            }
        else:
            # Slippage too high, abort
            return {
                'action': 'ABORT',
                'reason': 'Slippage exceeds acceptable threshold'
            }
    
    async def emergency_stop(self, state: BrainState):
        """Emergency stop all operations"""
        
        # Cancel all pending transactions
        for tx_hash in state.get('pending_transactions', []):
            try:
                await self.cancel_transaction(tx_hash)
            except:
                pass  # Best effort
        
        # Exit all risky positions
        for position in state.get('active_positions', []):
            if position['risk_score'] > 0.7:
                await self.exit_position(position)
        
        # Set brain to safe mode
        state['emergency_stop_active'] = True
        state['resume_time'] = datetime.now() + timedelta(hours=1)
        
        # Alert user
        await self.send_emergency_alert(state)
```

## Performance Optimization

### Caching Strategy

```python
class BrainCache:
    """Intelligent caching for brain operations"""
    
    def __init__(self):
        self.cache = {}
        self.ttls = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        
        if key in self.cache:
            if datetime.now() < self.ttls[key]:
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.ttls[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 60):
        """Cache value with TTL"""
        
        self.cache[key] = value
        self.ttls[key] = datetime.now() + timedelta(seconds=ttl)
    
    def cache_pool_data(self, pool_address: str, data: Dict):
        """Cache pool data with appropriate TTL"""
        
        # Static data - long TTL
        self.set(f"pool_static_{pool_address}", {
            'token0': data['token0'],
            'token1': data['token1'],
            'stable': data['stable']
        }, ttl=3600)  # 1 hour
        
        # Dynamic data - short TTL
        self.set(f"pool_dynamic_{pool_address}", {
            'reserves': data['reserves'],
            'apr': data['apr'],
            'volume': data['volume']
        }, ttl=60)  # 1 minute
```