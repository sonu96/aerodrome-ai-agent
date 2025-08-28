"""
Market observation node for the Aerodrome Brain.

This node handles comprehensive market data collection including:
- Pool data and analytics
- Wallet balances and positions
- Gas prices and network conditions
- Market sentiment indicators

Uses async operations for efficient data collection and includes
robust error handling and caching strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from ..state import BrainState, BrainConfig


class ObserverNode:
    """
    Market observation node that collects comprehensive market data.
    
    This node is responsible for gathering all necessary market information
    to inform the brain's decision-making process. It includes:
    
    - Aerodrome pool data collection
    - Portfolio state analysis  
    - Gas price and network monitoring
    - Market condition assessment
    """

    def __init__(self, cdp_manager, config: BrainConfig):
        """
        Initialize the observer node.
        
        Args:
            cdp_manager: CDP SDK manager for blockchain interactions
            config: Brain configuration parameters
        """
        self.cdp_manager = cdp_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cache for reducing API calls
        self._cache = {}
        self._cache_timestamps = {}
        
        # Pool addresses and constants
        self.FACTORY_ADDRESS = "0x420DD381b31aEf6683db6B902084cB0FFECe40Da"
        self.ROUTER_ADDRESS = "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
        
        # Rate limiting
        self._last_request_time = {}
        self._min_request_interval = 1.0  # 1 second between requests

    async def observe(self, state: BrainState) -> BrainState:
        """
        Main observation function that collects all market data.
        
        Args:
            state: Current brain state
            
        Returns:
            Updated state with market observations
        """
        
        self.logger.info("Starting market observation")
        
        # Collect data in parallel for efficiency
        tasks = [
            self._collect_pool_data(),
            self._collect_wallet_data(), 
            self._collect_gas_data(),
            self._collect_network_conditions(),
            self._collect_portfolio_positions(),
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            pool_data, wallet_data, gas_data, network_data, position_data = results
            
            # Handle any exceptions in the results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error in observation task {i}: {result}")
            
            # Calculate derived metrics
            total_tvl = self._calculate_total_tvl(pool_data)
            network_congestion = self._assess_network_congestion(gas_data, network_data)
            base_fee_trend = self._analyze_base_fee_trend(gas_data)
            
            # Update state with observations
            updated_state = {
                **state,
                'timestamp': datetime.now(),
                'market_data': self._process_pool_data(pool_data),
                'wallet_balance': wallet_data if not isinstance(wallet_data, Exception) else {},
                'active_positions': position_data if not isinstance(position_data, Exception) else [],
                'gas_price': gas_data.get('current_price', 0) if not isinstance(gas_data, Exception) else 0,
                'network_congestion': network_congestion,
                'base_fee_trend': base_fee_trend,
                'total_value_locked': total_tvl,
                'portfolio_performance': self._calculate_portfolio_performance(wallet_data, position_data),
                'market_sentiment': self._assess_market_sentiment(pool_data),
                'pool_analytics': self._generate_pool_analytics(pool_data)
            }
            
            self.logger.info(f"Market observation completed - found {len(pool_data or [])} pools")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Critical error in market observation: {e}")
            
            # Return state with minimal safe data
            return {
                **state,
                'timestamp': datetime.now(),
                'market_data': {},
                'gas_price': 0,
                'network_congestion': 1.0,  # Assume high congestion on error
                'warnings': state.get('warnings', []) + [f"Market observation failed: {str(e)}"]
            }

    async def _collect_pool_data(self) -> List[Dict[str, Any]]:
        """Collect comprehensive Aerodrome pool data."""
        
        # Check cache first
        cache_key = "pool_data"
        if self._is_cached_valid(cache_key, ttl=60):  # 1 minute cache
            return self._cache[cache_key]
        
        try:
            # Rate limiting
            await self._rate_limit("pool_data")
            
            pools = []
            
            # Get all pools from factory
            all_pools_data = await self.cdp_manager.read_contract(
                contract_address=self.FACTORY_ADDRESS,
                method="allPoolsLength",
                abi=self._get_factory_abi()
            )
            
            total_pools = int(all_pools_data) if all_pools_data else 0
            
            # Limit to top pools by TVL for performance (configurable)
            max_pools_to_analyze = min(total_pools, self.config.max_concurrent_analysis * 4)
            
            # Get pool addresses in batches
            pool_addresses = await self._get_pool_addresses_batch(
                0, max_pools_to_analyze
            )
            
            # Collect detailed data for each pool
            pool_tasks = [
                self._collect_single_pool_data(address) 
                for address in pool_addresses[:self.config.max_concurrent_analysis]
            ]
            
            pool_results = await asyncio.gather(*pool_tasks, return_exceptions=True)
            
            # Filter out exceptions and add valid pools
            for result in pool_results:
                if not isinstance(result, Exception) and result:
                    pools.append(result)
            
            # Cache the results
            self._cache[cache_key] = pools
            self._cache_timestamps[cache_key] = datetime.now()
            
            self.logger.info(f"Collected data for {len(pools)} pools")
            return pools
            
        except Exception as e:
            self.logger.error(f"Error collecting pool data: {e}")
            return []

    async def _collect_single_pool_data(self, pool_address: str) -> Optional[Dict[str, Any]]:
        """Collect comprehensive data for a single pool."""
        
        try:
            # Parallel contract reads for efficiency
            tasks = [
                self._get_pool_reserves(pool_address),
                self._get_pool_metadata(pool_address),
                self._get_pool_fees(pool_address),
                self._get_pool_volume_24h(pool_address),
                self._get_pool_apr(pool_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            reserves, metadata, fees, volume_24h, apr = results
            
            # Handle exceptions in individual results
            if isinstance(reserves, Exception):
                reserves = {'reserve0': 0, 'reserve1': 0}
            if isinstance(metadata, Exception):
                metadata = {}
            if isinstance(fees, Exception):
                fees = 0
            if isinstance(volume_24h, Exception):
                volume_24h = 0
            if isinstance(apr, Exception):
                apr = 0
            
            # Calculate derived metrics
            tvl = self._calculate_pool_tvl(reserves, metadata)
            liquidity_depth = self._calculate_liquidity_depth(reserves)
            volatility = await self._calculate_pool_volatility(pool_address)
            
            return {
                'address': pool_address,
                'reserves': reserves,
                'metadata': metadata,
                'fees': fees,
                'volume_24h': volume_24h,
                'apr': apr,
                'tvl': tvl,
                'liquidity_depth': liquidity_depth,
                'volatility': volatility,
                'last_updated': datetime.now(),
                'is_stable': metadata.get('stable', False),
                'token0_symbol': metadata.get('token0_symbol', 'UNKNOWN'),
                'token1_symbol': metadata.get('token1_symbol', 'UNKNOWN')
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting data for pool {pool_address}: {e}")
            return None

    async def _collect_wallet_data(self) -> Dict[str, float]:
        """Collect current wallet balances for all relevant tokens."""
        
        cache_key = "wallet_data"
        if self._is_cached_valid(cache_key, ttl=30):  # 30 second cache
            return self._cache[cache_key]
        
        try:
            await self._rate_limit("wallet_data")
            
            if not self.cdp_manager or not hasattr(self.cdp_manager, 'wallet'):
                return {}
            
            wallet = self.cdp_manager.wallet
            wallet_address = wallet.default_address.address_id
            
            # Get all token balances
            balances = {}
            
            # Get native ETH balance
            eth_balance = await wallet.balance("eth")
            balances['eth'] = float(eth_balance.amount) if eth_balance else 0.0
            
            # Get common token balances (USDC, USDT, WETH, etc.)
            common_tokens = {
                'usdc': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
                'usdt': '0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2', 
                'weth': '0x4200000000000000000000000000000000000006',
                'aerodrome': '0x940181a94A35A4569E4529A3CDfB74e38FD98631'
            }
            
            for symbol, address in common_tokens.items():
                try:
                    balance = await wallet.balance(symbol)
                    balances[address.lower()] = float(balance.amount) if balance else 0.0
                except:
                    balances[address.lower()] = 0.0
            
            # Cache results
            self._cache[cache_key] = balances
            self._cache_timestamps[cache_key] = datetime.now()
            
            return balances
            
        except Exception as e:
            self.logger.error(f"Error collecting wallet data: {e}")
            return {}

    async def _collect_gas_data(self) -> Dict[str, Any]:
        """Collect current gas price and fee data."""
        
        cache_key = "gas_data"
        if self._is_cached_valid(cache_key, ttl=15):  # 15 second cache
            return self._cache[cache_key]
        
        try:
            await self._rate_limit("gas_data")
            
            # Get current gas price from network
            # This would typically use the CDP SDK or web3 provider
            gas_data = {
                'current_price': await self._get_current_gas_price(),
                'base_fee': await self._get_base_fee(),
                'priority_fee': await self._get_priority_fee(),
                'timestamp': datetime.now()
            }
            
            # Add historical trend data
            gas_data['trend'] = await self._analyze_gas_trend()
            
            # Cache results
            self._cache[cache_key] = gas_data
            self._cache_timestamps[cache_key] = datetime.now()
            
            return gas_data
            
        except Exception as e:
            self.logger.error(f"Error collecting gas data: {e}")
            return {
                'current_price': 0,
                'base_fee': 0,
                'priority_fee': 0,
                'trend': 'unknown',
                'timestamp': datetime.now()
            }

    async def _collect_network_conditions(self) -> Dict[str, Any]:
        """Assess current network conditions and congestion."""
        
        try:
            # This would implement actual network monitoring
            # For now, return mock data structure
            return {
                'block_time': 2.0,  # Average block time in seconds
                'pending_transactions': 1000,  # Estimated pending txs
                'congestion_level': 'normal',  # low, normal, high, critical
                'recommended_gas_multiplier': 1.1
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing network conditions: {e}")
            return {
                'block_time': 3.0,
                'pending_transactions': 0,
                'congestion_level': 'unknown',
                'recommended_gas_multiplier': 1.5
            }

    async def _collect_portfolio_positions(self) -> List[Dict[str, Any]]:
        """Collect information about active liquidity positions."""
        
        try:
            if not self.cdp_manager:
                return []
            
            # This would implement actual position tracking
            # For now, return empty list as placeholder
            positions = []
            
            # TODO: Implement actual position collection
            # - Query user's LP positions
            # - Get position values and performance
            # - Calculate impermanent loss
            # - Track rewards and emissions
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error collecting portfolio positions: {e}")
            return []

    # Helper methods for data processing

    def _process_pool_data(self, pools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw pool data into structured market data."""
        
        if not pools:
            return {}
        
        processed = {
            'pools': {pool['address']: pool for pool in pools},
            'total_pools': len(pools),
            'total_tvl': sum(pool.get('tvl', 0) for pool in pools),
            'average_apr': np.mean([pool.get('apr', 0) for pool in pools]),
            'high_volume_pools': [
                pool for pool in pools 
                if pool.get('volume_24h', 0) > 10000
            ],
            'stable_pools': [
                pool for pool in pools
                if pool.get('is_stable', False)
            ],
            'volatile_pools': [
                pool for pool in pools
                if not pool.get('is_stable', False)
            ]
        }
        
        return processed

    def _calculate_total_tvl(self, pool_data: List[Dict[str, Any]]) -> float:
        """Calculate total TVL across all pools."""
        
        if not pool_data or isinstance(pool_data, Exception):
            return 0.0
        
        return sum(pool.get('tvl', 0) for pool in pool_data)

    def _assess_network_congestion(
        self, 
        gas_data: Dict[str, Any], 
        network_data: Dict[str, Any]
    ) -> float:
        """Assess network congestion level (0.0 to 1.0)."""
        
        try:
            if isinstance(gas_data, Exception) or isinstance(network_data, Exception):
                return 1.0  # Assume high congestion on error
            
            # Base congestion on gas price and block time
            current_gas = gas_data.get('current_price', 0)
            block_time = network_data.get('block_time', 2.0)
            
            # Normal gas price on Base is around 0.001 gwei
            gas_factor = min(current_gas / 0.01, 1.0)  # Normalize to 0.01 gwei
            
            # Normal block time is 2 seconds
            time_factor = max((block_time - 2.0) / 3.0, 0.0)  # Penalty for slow blocks
            
            congestion = (gas_factor + time_factor) / 2.0
            return min(congestion, 1.0)
            
        except:
            return 0.5  # Default moderate congestion

    def _analyze_base_fee_trend(self, gas_data: Dict[str, Any]) -> str:
        """Analyze base fee trend direction."""
        
        try:
            if isinstance(gas_data, Exception):
                return "unknown"
            
            trend = gas_data.get('trend', 'stable')
            return trend
            
        except:
            return "stable"

    def _calculate_portfolio_performance(
        self, 
        wallet_data: Dict[str, float], 
        position_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        
        try:
            if isinstance(wallet_data, Exception):
                wallet_data = {}
            if isinstance(position_data, Exception):
                position_data = []
            
            # Calculate total portfolio value
            wallet_value = sum(wallet_data.values())
            position_value = sum(pos.get('value_usd', 0) for pos in position_data)
            total_value = wallet_value + position_value
            
            return {
                'total_value': total_value,
                'wallet_value': wallet_value,
                'position_value': position_value,
                'position_count': len(position_data),
                'last_updated': datetime.now().timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio performance: {e}")
            return {
                'total_value': 0,
                'wallet_value': 0,
                'position_value': 0,
                'position_count': 0,
                'last_updated': datetime.now().timestamp()
            }

    def _assess_market_sentiment(self, pool_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall market sentiment based on pool data."""
        
        try:
            if not pool_data or isinstance(pool_data, Exception):
                return {'sentiment': 'neutral', 'confidence': 0.0}
            
            # Analyze volume trends
            total_volume = sum(pool.get('volume_24h', 0) for pool in pool_data)
            high_apr_pools = [p for p in pool_data if p.get('apr', 0) > 20]
            
            # Simple sentiment calculation
            if len(high_apr_pools) > len(pool_data) * 0.3:
                sentiment = 'bullish'
                confidence = 0.7
            elif total_volume > 1_000_000:  # High volume
                sentiment = 'active'
                confidence = 0.6
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'volume_24h': total_volume,
                'high_apr_count': len(high_apr_pools)
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing market sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}

    def _generate_pool_analytics(self, pool_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate detailed analytics for each pool."""
        
        analytics = {}
        
        try:
            if not pool_data or isinstance(pool_data, Exception):
                return analytics
            
            for pool in pool_data:
                pool_addr = pool['address']
                analytics[pool_addr] = {
                    'score': self._calculate_pool_score(pool),
                    'risk_level': self._assess_pool_risk(pool),
                    'liquidity_rating': self._rate_liquidity(pool),
                    'volume_trend': self._analyze_volume_trend(pool),
                    'apr_stability': self._assess_apr_stability(pool)
                }
            
        except Exception as e:
            self.logger.error(f"Error generating pool analytics: {e}")
        
        return analytics

    # Cache and rate limiting helpers

    def _is_cached_valid(self, key: str, ttl: int) -> bool:
        """Check if cached data is still valid."""
        
        if key not in self._cache or key not in self._cache_timestamps:
            return False
        
        age = (datetime.now() - self._cache_timestamps[key]).total_seconds()
        return age < ttl

    async def _rate_limit(self, operation: str):
        """Rate limit API calls to prevent overwhelming the network."""
        
        if operation in self._last_request_time:
            elapsed = time.time() - self._last_request_time[operation]
            if elapsed < self._min_request_interval:
                wait_time = self._min_request_interval - elapsed
                await asyncio.sleep(wait_time)
        
        self._last_request_time[operation] = time.time()

    # Placeholder methods for actual blockchain interactions
    # These would be implemented with real CDP SDK calls

    async def _get_pool_addresses_batch(self, start: int, count: int) -> List[str]:
        """Get pool addresses in batch."""
        # Placeholder - would implement actual factory contract calls
        return []

    async def _get_pool_reserves(self, pool_address: str) -> Dict[str, float]:
        """Get pool reserves."""
        # Placeholder
        return {'reserve0': 0, 'reserve1': 0}

    async def _get_pool_metadata(self, pool_address: str) -> Dict[str, Any]:
        """Get pool metadata."""
        # Placeholder
        return {}

    async def _get_pool_fees(self, pool_address: str) -> float:
        """Get pool fees."""
        # Placeholder
        return 0.0

    async def _get_pool_volume_24h(self, pool_address: str) -> float:
        """Get 24h volume."""
        # Placeholder
        return 0.0

    async def _get_pool_apr(self, pool_address: str) -> float:
        """Get pool APR."""
        # Placeholder
        return 0.0

    async def _get_current_gas_price(self) -> float:
        """Get current gas price."""
        # Placeholder
        return 0.001

    async def _get_base_fee(self) -> float:
        """Get current base fee."""
        # Placeholder
        return 0.001

    async def _get_priority_fee(self) -> float:
        """Get current priority fee."""
        # Placeholder
        return 0.0

    async def _analyze_gas_trend(self) -> str:
        """Analyze gas price trend."""
        # Placeholder
        return "stable"

    def _calculate_pool_tvl(self, reserves: Dict[str, float], metadata: Dict[str, Any]) -> float:
        """Calculate pool TVL from reserves."""
        # Placeholder
        return 0.0

    def _calculate_liquidity_depth(self, reserves: Dict[str, float]) -> float:
        """Calculate liquidity depth."""
        # Placeholder
        return 0.0

    async def _calculate_pool_volatility(self, pool_address: str) -> float:
        """Calculate pool volatility."""
        # Placeholder
        return 0.0

    def _calculate_pool_score(self, pool: Dict[str, Any]) -> float:
        """Calculate pool opportunity score."""
        # Placeholder
        return 0.5

    def _assess_pool_risk(self, pool: Dict[str, Any]) -> str:
        """Assess pool risk level."""
        # Placeholder
        return "medium"

    def _rate_liquidity(self, pool: Dict[str, Any]) -> str:
        """Rate pool liquidity."""
        # Placeholder
        return "good"

    def _analyze_volume_trend(self, pool: Dict[str, Any]) -> str:
        """Analyze volume trend."""
        # Placeholder
        return "stable"

    def _assess_apr_stability(self, pool: Dict[str, Any]) -> str:
        """Assess APR stability."""
        # Placeholder
        return "stable"

    def _get_factory_abi(self) -> List[Dict[str, Any]]:
        """Get factory contract ABI."""
        # Placeholder - would return actual ABI
        return []