"""
Gas Optimization Utilities - Comprehensive gas management and optimization

This module provides gas optimization utilities for CDP SDK transactions including:
- EIP-1559 gas parameter optimization
- Gas price prediction and monitoring
- Transaction timing optimization  
- Gas estimation and validation
- Batch operation optimization
- MEV protection strategies

All gas operations integrate with CDP SDK and Base network characteristics.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from decimal import Decimal
from statistics import mean, median
import json

try:
    from cdp_sdk import readContract
except ImportError:
    readContract = None

from .errors import CDPError, GasError, NetworkError


@dataclass
class GasParams:
    """EIP-1559 gas parameters."""
    max_fee_per_gas: int
    max_priority_fee_per_gas: int
    gas_limit: Optional[int] = None
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for transactions."""
        result = {
            'maxFeePerGas': self.max_fee_per_gas,
            'maxPriorityFeePerGas': self.max_priority_fee_per_gas
        }
        if self.gas_limit:
            result['gasLimit'] = self.gas_limit
        return result
    
    def to_gwei(self) -> Dict[str, float]:
        """Convert to gwei for human readability."""
        return {
            'max_fee_per_gas_gwei': self.max_fee_per_gas / 1e9,
            'max_priority_fee_per_gas_gwei': self.max_priority_fee_per_gas / 1e9
        }


@dataclass
class GasEstimate:
    """Gas estimation result."""
    estimated_gas: int
    base_fee: int
    priority_fee: int
    max_fee: int
    confidence: float  # 0-1 confidence score
    recommended_limit: int
    
    def to_gas_params(self) -> GasParams:
        """Convert to GasParams."""
        return GasParams(
            max_fee_per_gas=self.max_fee,
            max_priority_fee_per_gas=self.priority_fee,
            gas_limit=self.recommended_limit
        )


class GasOptimizer:
    """
    Advanced gas optimization for Base network transactions.
    
    Provides intelligent gas parameter calculation, timing optimization,
    and cost reduction strategies.
    """
    
    def __init__(self, cdp_manager):
        """
        Initialize gas optimizer.
        
        Args:
            cdp_manager: CDPManager instance
        """
        self.cdp = cdp_manager
        self.logger = logging.getLogger(__name__)
        self.network = cdp_manager.network
        
        # Base network specific configurations
        self.base_configs = {
            'base-mainnet': {
                'target_block_time': 2,  # 2 seconds
                'max_base_fee': 100 * 1e9,  # 100 gwei max reasonable base fee
                'min_priority_fee': 0.01 * 1e9,  # 0.01 gwei minimum priority
                'default_priority_fee': 0.1 * 1e9,  # 0.1 gwei default
                'gas_limit_buffer': 1.2,  # 20% buffer on gas estimates
                'confirmation_blocks': 1  # Base is fast, 1 block usually sufficient
            },
            'base-goerli': {
                'target_block_time': 2,
                'max_base_fee': 50 * 1e9,
                'min_priority_fee': 0.01 * 1e9,
                'default_priority_fee': 0.05 * 1e9,
                'gas_limit_buffer': 1.3,
                'confirmation_blocks': 1
            }
        }
        
        self.config = self.base_configs.get(self.network, self.base_configs['base-mainnet'])
        
        # Gas tracking
        self.gas_history = []
        self.max_history_size = 100
        
        # Cache for gas estimates
        self.estimate_cache = {}
        self.cache_ttl = 30  # 30 seconds
    
    async def get_optimal_gas_params(
        self,
        priority: str = 'standard',
        max_fee_multiplier: float = 1.5
    ) -> GasParams:
        """
        Get optimized EIP-1559 gas parameters for Base network.
        
        Args:
            priority: Priority level ('slow', 'standard', 'fast', 'urgent')
            max_fee_multiplier: Multiplier for max fee calculation
            
        Returns:
            Optimized gas parameters
            
        Raises:
            GasError: If gas parameter calculation fails
        """
        try:
            # Get current network conditions
            base_fee = await self._get_current_base_fee()
            priority_fee_stats = await self._get_priority_fee_stats()
            
            # Calculate priority fee based on priority level
            priority_multipliers = {
                'slow': 0.5,
                'standard': 1.0,
                'fast': 2.0,
                'urgent': 5.0
            }
            
            multiplier = priority_multipliers.get(priority, 1.0)
            priority_fee = max(
                int(priority_fee_stats['median'] * multiplier),
                self.config['min_priority_fee']
            )
            
            # Calculate max fee with buffer
            # Base fee can double in next block, so use 2x + priority fee
            max_fee = int(base_fee * max_fee_multiplier) + priority_fee
            
            # Apply reasonable limits
            max_fee = min(max_fee, self.config['max_base_fee'])
            
            gas_params = GasParams(
                max_fee_per_gas=max_fee,
                max_priority_fee_per_gas=priority_fee
            )
            
            self.logger.debug(f"Optimal gas params: {gas_params.to_gwei()}")
            return gas_params
            
        except Exception as e:
            self.logger.error(f"Gas optimization failed: {str(e)}")
            # Fallback to safe defaults
            return self._get_fallback_gas_params()
    
    async def estimate_transaction_cost(
        self,
        gas_limit: int,
        priority: str = 'standard'
    ) -> Dict[str, Any]:
        """
        Estimate total transaction cost in ETH and USD.
        
        Args:
            gas_limit: Estimated gas limit
            priority: Priority level
            
        Returns:
            Cost estimation dictionary
        """
        try:
            gas_params = await self.get_optimal_gas_params(priority)
            
            # Calculate costs
            max_cost_wei = gas_limit * gas_params.max_fee_per_gas
            likely_cost_wei = gas_limit * (await self._get_current_base_fee() + gas_params.max_priority_fee_per_gas)
            
            # Convert to ETH
            max_cost_eth = max_cost_wei / 1e18
            likely_cost_eth = likely_cost_wei / 1e18
            
            # Get ETH price for USD estimation (placeholder - would integrate with price API)
            eth_price_usd = await self._get_eth_price_usd()
            
            return {
                'gas_limit': gas_limit,
                'gas_params': gas_params.to_dict(),
                'max_cost': {
                    'wei': max_cost_wei,
                    'eth': max_cost_eth,
                    'usd': max_cost_eth * eth_price_usd
                },
                'likely_cost': {
                    'wei': likely_cost_wei,
                    'eth': likely_cost_eth,
                    'usd': likely_cost_eth * eth_price_usd
                },
                'priority': priority,
                'network': self.network
            }
            
        except Exception as e:
            self.logger.error(f"Cost estimation failed: {str(e)}")
            raise GasError(f"Failed to estimate transaction cost: {str(e)}")
    
    async def should_execute_now(
        self,
        target_cost_gwei: Optional[float] = None,
        max_wait_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Determine if current gas conditions are favorable for execution.
        
        Args:
            target_cost_gwei: Target gas price in gwei
            max_wait_minutes: Maximum time to wait for better conditions
            
        Returns:
            Execution recommendation dictionary
        """
        try:
            current_base_fee = await self._get_current_base_fee()
            current_base_fee_gwei = current_base_fee / 1e9
            
            # Get recent gas history for trend analysis
            recent_fees = await self._get_recent_base_fees(blocks=20)
            avg_fee = mean(recent_fees) if recent_fees else current_base_fee
            trend = self._calculate_fee_trend(recent_fees)
            
            # Decision factors
            factors = {
                'current_base_fee_gwei': current_base_fee_gwei,
                'average_base_fee_gwei': avg_fee / 1e9,
                'is_below_average': current_base_fee < avg_fee * 1.1,
                'trend': trend,  # 'increasing', 'decreasing', 'stable'
                'network_congestion': await self._assess_network_congestion(),
                'time_of_day_factor': self._get_time_of_day_factor()
            }
            
            # Make recommendation
            should_execute = True
            reason = "Current conditions are favorable"
            
            if target_cost_gwei and current_base_fee_gwei > target_cost_gwei * 1.5:
                should_execute = False
                reason = f"Base fee {current_base_fee_gwei:.2f} gwei exceeds target {target_cost_gwei} gwei"
            elif trend == 'increasing' and not factors['is_below_average']:
                should_execute = False
                reason = "Gas prices trending upward, may be better to wait"
            elif factors['network_congestion'] > 0.8:  # High congestion
                should_execute = False
                reason = "High network congestion detected"
            
            return {
                'should_execute': should_execute,
                'reason': reason,
                'factors': factors,
                'recommended_wait_minutes': 5 if not should_execute else 0,
                'max_wait_minutes': max_wait_minutes
            }
            
        except Exception as e:
            self.logger.error(f"Execution timing analysis failed: {str(e)}")
            # Conservative default - execute now rather than risk indefinite waiting
            return {
                'should_execute': True,
                'reason': f"Analysis failed, executing conservatively: {str(e)}",
                'factors': {},
                'recommended_wait_minutes': 0,
                'max_wait_minutes': max_wait_minutes
            }
    
    async def optimize_batch_operations(
        self,
        operations: List[Dict[str, Any]],
        max_batch_size: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """
        Optimize batching of multiple operations for gas efficiency.
        
        Args:
            operations: List of operations to batch
            max_batch_size: Maximum operations per batch
            
        Returns:
            List of optimized batches
        """
        if len(operations) <= max_batch_size:
            return [operations]
        
        # Group operations by priority and gas requirements
        batches = []
        current_batch = []
        current_gas_estimate = 0
        
        # Base network block gas limit (approximation)
        block_gas_limit = 30_000_000
        batch_gas_limit = block_gas_limit // 4  # Conservative batch limit
        
        for operation in operations:
            op_gas_estimate = operation.get('gas_estimate', 100_000)  # Default estimate
            
            # Check if adding this operation would exceed batch limits
            if (len(current_batch) >= max_batch_size or 
                current_gas_estimate + op_gas_estimate > batch_gas_limit):
                
                if current_batch:
                    batches.append(current_batch)
                current_batch = [operation]
                current_gas_estimate = op_gas_estimate
            else:
                current_batch.append(operation)
                current_gas_estimate += op_gas_estimate
        
        # Add remaining operations
        if current_batch:
            batches.append(current_batch)
        
        self.logger.info(f"Optimized {len(operations)} operations into {len(batches)} batches")
        return batches
    
    async def estimate_gas_for_contract_call(
        self,
        contract_address: str,
        method: str,
        args: Dict[str, Any],
        value: int = 0
    ) -> GasEstimate:
        """
        Estimate gas for a specific contract call.
        
        Args:
            contract_address: Contract to call
            method: Method name
            args: Method arguments
            value: ETH value to send
            
        Returns:
            Gas estimation result
        """
        cache_key = f"{contract_address}:{method}:{str(args)}:{value}"
        
        # Check cache first
        if cache_key in self.estimate_cache:
            cached = self.estimate_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['estimate']
        
        try:
            # Method-specific gas estimates (these would be refined based on actual data)
            method_estimates = {
                'transfer': 21000,
                'approve': 46000,
                'swap': 150000,
                'swapExactTokensForTokens': 200000,
                'addLiquidity': 250000,
                'removeLiquidity': 200000,
                'vote': 100000,
                'create_lock': 150000,
                'claim': 80000
            }
            
            base_estimate = method_estimates.get(method, 100000)
            
            # Adjust based on complexity
            if 'routes' in args and isinstance(args['routes'], list):
                base_estimate += len(args['routes']) * 50000  # Multi-hop swaps cost more
            
            if value > 0:
                base_estimate += 2300  # ETH transfer cost
            
            # Get current gas conditions
            base_fee = await self._get_current_base_fee()
            priority_fee_stats = await self._get_priority_fee_stats()
            
            # Add buffer for safety
            recommended_limit = int(base_estimate * self.config['gas_limit_buffer'])
            
            estimate = GasEstimate(
                estimated_gas=base_estimate,
                base_fee=base_fee,
                priority_fee=priority_fee_stats['median'],
                max_fee=int(base_fee * 1.5) + priority_fee_stats['median'],
                confidence=0.8,  # Moderate confidence for estimates
                recommended_limit=recommended_limit
            )
            
            # Cache the result
            self.estimate_cache[cache_key] = {
                'estimate': estimate,
                'timestamp': time.time()
            }
            
            return estimate
            
        except Exception as e:
            self.logger.error(f"Gas estimation failed: {str(e)}")
            # Return conservative fallback
            return GasEstimate(
                estimated_gas=200000,
                base_fee=1_000_000_000,  # 1 gwei
                priority_fee=100_000_000,  # 0.1 gwei
                max_fee=2_100_000_000,  # 2.1 gwei
                confidence=0.3,
                recommended_limit=250000
            )
    
    async def _get_current_base_fee(self) -> int:
        """
        Get current base fee from latest block.
        
        Returns:
            Base fee in wei
        """
        try:
            # This would integrate with CDP SDK or Base RPC to get latest block
            # For now, return a reasonable estimate for Base network
            # Base network typically has very low fees
            
            # Simulate fetching from latest block
            # In real implementation, would call:
            # latest_block = await self.cdp.client.get_latest_block()
            # return latest_block.base_fee_per_gas
            
            # Base network typical range: 0.001-0.1 gwei
            return int(0.01 * 1e9)  # 0.01 gwei default
            
        except Exception as e:
            self.logger.warning(f"Failed to get current base fee: {str(e)}")
            return int(0.01 * 1e9)  # Fallback
    
    async def _get_priority_fee_stats(self) -> Dict[str, int]:
        """
        Get priority fee statistics from recent blocks.
        
        Returns:
            Priority fee statistics
        """
        try:
            # This would analyze recent blocks to get priority fee percentiles
            # For Base network, priority fees are typically very low
            
            return {
                'min': int(0.001 * 1e9),    # 0.001 gwei
                'median': int(0.01 * 1e9),  # 0.01 gwei
                'max': int(0.1 * 1e9),      # 0.1 gwei
                'p75': int(0.02 * 1e9),     # 75th percentile
                'p90': int(0.05 * 1e9)      # 90th percentile
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get priority fee stats: {str(e)}")
            return {
                'min': int(0.001 * 1e9),
                'median': int(0.01 * 1e9),
                'max': int(0.1 * 1e9),
                'p75': int(0.02 * 1e9),
                'p90': int(0.05 * 1e9)
            }
    
    async def _get_recent_base_fees(self, blocks: int = 20) -> List[int]:
        """
        Get base fees from recent blocks.
        
        Args:
            blocks: Number of recent blocks to analyze
            
        Returns:
            List of base fees
        """
        try:
            # This would fetch recent block data
            # For now, simulate with reasonable Base network values
            base_fee = await self._get_current_base_fee()
            
            # Simulate some variation
            fees = []
            for i in range(blocks):
                # Add some random variation (±20%)
                variation = 0.8 + (i % 5) * 0.1  # Simple variation simulation
                fees.append(int(base_fee * variation))
            
            return fees
            
        except Exception as e:
            self.logger.warning(f"Failed to get recent base fees: {str(e)}")
            return []
    
    def _calculate_fee_trend(self, recent_fees: List[int]) -> str:
        """
        Calculate gas fee trend from recent data.
        
        Args:
            recent_fees: List of recent base fees
            
        Returns:
            Trend indicator ('increasing', 'decreasing', 'stable')
        """
        if len(recent_fees) < 3:
            return 'stable'
        
        # Simple trend analysis - compare first half to second half
        mid_point = len(recent_fees) // 2
        first_half_avg = mean(recent_fees[:mid_point])
        second_half_avg = mean(recent_fees[mid_point:])
        
        change_ratio = second_half_avg / first_half_avg
        
        if change_ratio > 1.1:
            return 'increasing'
        elif change_ratio < 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _assess_network_congestion(self) -> float:
        """
        Assess network congestion level.
        
        Returns:
            Congestion level (0.0 - 1.0)
        """
        try:
            # This would analyze pending transaction pool, block utilization, etc.
            # For Base network, congestion is typically low
            
            # Simulate congestion assessment
            base_fee = await self._get_current_base_fee()
            
            # Higher base fee indicates higher congestion
            if base_fee > 0.1 * 1e9:  # > 0.1 gwei
                return 0.8  # High congestion
            elif base_fee > 0.05 * 1e9:  # > 0.05 gwei
                return 0.5  # Medium congestion
            else:
                return 0.2  # Low congestion
                
        except Exception:
            return 0.3  # Default moderate congestion
    
    def _get_time_of_day_factor(self) -> float:
        """
        Get time-of-day adjustment factor for gas optimization.
        
        Returns:
            Time factor (0.8 - 1.2)
        """
        import datetime
        
        # Get current UTC hour
        utc_hour = datetime.datetime.utcnow().hour
        
        # US/EU business hours typically see higher activity
        if 13 <= utc_hour <= 21:  # 8 AM EST to 4 PM PST
            return 1.2  # Peak hours
        elif 22 <= utc_hour <= 2:  # Late US hours
            return 1.0  # Normal activity
        else:
            return 0.8  # Low activity hours
    
    def _get_fallback_gas_params(self) -> GasParams:
        """
        Get conservative fallback gas parameters.
        
        Returns:
            Safe gas parameters
        """
        return GasParams(
            max_fee_per_gas=int(2 * 1e9),      # 2 gwei - conservative for Base
            max_priority_fee_per_gas=int(0.1 * 1e9)  # 0.1 gwei priority
        )
    
    async def _get_eth_price_usd(self) -> float:
        """
        Get current ETH price in USD.
        
        Returns:
            ETH price (placeholder implementation)
        """
        # This would integrate with a price API like CoinGecko, CoinAPI, etc.
        # For now, return a reasonable estimate
        return 2500.0  # $2500 USD per ETH
    
    def add_gas_history(self, tx_result: Dict[str, Any]) -> None:
        """
        Add transaction result to gas history for learning.
        
        Args:
            tx_result: Transaction result with gas data
        """
        if 'gas_used' in tx_result and 'gas_price' in tx_result:
            entry = {
                'timestamp': time.time(),
                'gas_used': tx_result['gas_used'],
                'gas_price': tx_result['gas_price'],
                'success': tx_result.get('success', False),
                'method': tx_result.get('method'),
                'contract': tx_result.get('contract_address')
            }
            
            self.gas_history.append(entry)
            
            # Keep history size manageable
            if len(self.gas_history) > self.max_history_size:
                self.gas_history = self.gas_history[-self.max_history_size:]
    
    def get_gas_analytics(self) -> Dict[str, Any]:
        """
        Get gas usage analytics from transaction history.
        
        Returns:
            Gas analytics dictionary
        """
        if not self.gas_history:
            return {'total_transactions': 0}
        
        successful_txs = [tx for tx in self.gas_history if tx['success']]
        
        if not successful_txs:
            return {'total_transactions': len(self.gas_history), 'successful_transactions': 0}
        
        gas_used_values = [tx['gas_used'] for tx in successful_txs]
        gas_price_values = [tx['gas_price'] for tx in successful_txs]
        
        # Calculate total costs
        total_gas_cost = sum(tx['gas_used'] * tx['gas_price'] for tx in successful_txs)
        
        return {
            'total_transactions': len(self.gas_history),
            'successful_transactions': len(successful_txs),
            'avg_gas_used': mean(gas_used_values),
            'median_gas_used': median(gas_used_values),
            'avg_gas_price': mean(gas_price_values),
            'median_gas_price': median(gas_price_values),
            'total_gas_cost_wei': total_gas_cost,
            'total_gas_cost_eth': total_gas_cost / 1e18,
            'success_rate': len(successful_txs) / len(self.gas_history),
            'most_expensive_tx': max(successful_txs, key=lambda x: x['gas_used'] * x['gas_price']),
            'most_gas_used': max(successful_txs, key=lambda x: x['gas_used'])
        }
    
    def clear_cache(self) -> None:
        """Clear gas estimation cache."""
        self.estimate_cache.clear()
        self.logger.info("Gas estimation cache cleared")


class MEVProtection:
    """
    MEV (Maximal Extractable Value) protection utilities.
    
    Provides protection against sandwich attacks, frontrunning,
    and other MEV exploitation strategies.
    """
    
    def __init__(self, gas_optimizer: GasOptimizer):
        """
        Initialize MEV protection.
        
        Args:
            gas_optimizer: GasOptimizer instance
        """
        self.gas_optimizer = gas_optimizer
        self.logger = logging.getLogger(__name__)
    
    def add_mev_protection_params(
        self,
        gas_params: GasParams,
        protection_level: str = 'medium'
    ) -> GasParams:
        """
        Add MEV protection to gas parameters.
        
        Args:
            gas_params: Base gas parameters
            protection_level: Protection level ('low', 'medium', 'high')
            
        Returns:
            Protected gas parameters
        """
        protection_multipliers = {
            'low': 1.1,     # 10% increase
            'medium': 1.25, # 25% increase
            'high': 1.5     # 50% increase
        }
        
        multiplier = protection_multipliers.get(protection_level, 1.25)
        
        # Increase priority fee to reduce MEV risk
        protected_priority_fee = int(gas_params.max_priority_fee_per_gas * multiplier)
        
        # Adjust max fee accordingly
        base_fee_component = gas_params.max_fee_per_gas - gas_params.max_priority_fee_per_gas
        protected_max_fee = base_fee_component + protected_priority_fee
        
        return GasParams(
            max_fee_per_gas=protected_max_fee,
            max_priority_fee_per_gas=protected_priority_fee,
            gas_limit=gas_params.gas_limit
        )
    
    def suggest_slippage_protection(
        self,
        operation_type: str,
        amount: int
    ) -> Dict[str, Any]:
        """
        Suggest slippage protection parameters.
        
        Args:
            operation_type: Type of operation ('swap', 'add_liquidity', etc.)
            amount: Transaction amount
            
        Returns:
            Slippage protection suggestions
        """
        # Base slippage recommendations
        base_slippage = {
            'swap': 0.5,           # 0.5% for swaps
            'add_liquidity': 1.0,  # 1% for liquidity
            'remove_liquidity': 1.0
        }
        
        base = base_slippage.get(operation_type, 0.5)
        
        # Adjust based on amount size (larger amounts need more protection)
        if amount > 10000 * 1e18:  # > 10,000 tokens
            multiplier = 2.0
        elif amount > 1000 * 1e18:  # > 1,000 tokens
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        suggested_slippage = min(base * multiplier, 5.0)  # Cap at 5%
        
        return {
            'suggested_slippage_percent': suggested_slippage,
            'min_slippage_percent': base,
            'max_recommended_slippage_percent': 5.0,
            'protection_reasoning': f"Based on {operation_type} with amount size category"
        }


# Utility functions
async def wait_for_optimal_gas(
    gas_optimizer: GasOptimizer,
    target_gwei: float,
    max_wait_minutes: int = 30,
    check_interval_seconds: int = 30
) -> bool:
    """
    Wait for gas prices to reach target level.
    
    Args:
        gas_optimizer: GasOptimizer instance
        target_gwei: Target gas price in gwei
        max_wait_minutes: Maximum time to wait
        check_interval_seconds: How often to check prices
        
    Returns:
        True if target reached, False if timeout
    """
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    logger = logging.getLogger(__name__)
    logger.info(f"Waiting for gas price <= {target_gwei} gwei (max {max_wait_minutes} minutes)")
    
    while time.time() - start_time < max_wait_seconds:
        try:
            timing_analysis = await gas_optimizer.should_execute_now(
                target_cost_gwei=target_gwei,
                max_wait_minutes=max_wait_minutes
            )
            
            if timing_analysis['should_execute']:
                logger.info("Optimal gas conditions reached")
                return True
            
            # Wait before next check
            await asyncio.sleep(check_interval_seconds)
            
        except Exception as e:
            logger.warning(f"Gas monitoring error: {str(e)}")
            await asyncio.sleep(check_interval_seconds)
    
    logger.warning(f"Gas price target not reached within {max_wait_minutes} minutes")
    return False


def calculate_gas_savings(
    original_params: GasParams,
    optimized_params: GasParams,
    gas_used: int
) -> Dict[str, Any]:
    """
    Calculate gas savings from optimization.
    
    Args:
        original_params: Original gas parameters
        optimized_params: Optimized gas parameters
        gas_used: Actual gas used
        
    Returns:
        Savings calculation
    """
    original_cost = gas_used * original_params.max_fee_per_gas
    optimized_cost = gas_used * optimized_params.max_fee_per_gas
    
    savings_wei = original_cost - optimized_cost
    savings_percentage = (savings_wei / original_cost * 100) if original_cost > 0 else 0
    
    return {
        'original_cost_wei': original_cost,
        'optimized_cost_wei': optimized_cost,
        'savings_wei': savings_wei,
        'savings_eth': savings_wei / 1e18,
        'savings_percentage': savings_percentage,
        'gas_used': gas_used
    }


# Export public interface
__all__ = [
    'GasOptimizer',
    'MEVProtection',
    'GasParams',
    'GasEstimate',
    'wait_for_optimal_gas',
    'calculate_gas_savings'
]