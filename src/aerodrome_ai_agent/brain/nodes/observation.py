"""
Observation Node - Market data collection and portfolio monitoring

Handles collection of market data, wallet balances, active positions,
and other environmental information needed for decision making.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

from ..state import BrainState
from ..config import BrainConfig


logger = logging.getLogger(__name__)


class ObservationNode:
    """
    Observation node for collecting market and portfolio data
    """
    
    def __init__(self, cdp_manager: Optional[Any], config: BrainConfig):
        self.cdp_manager = cdp_manager
        self.config = config
    
    async def execute(self, state: BrainState) -> BrainState:
        """Execute observation logic"""
        
        logger.info("Starting market observation")
        
        try:
            # Update timestamp
            state["timestamp"] = datetime.now()
            
            # Collect market data
            if self.cdp_manager:
                # Get real market data
                market_data = await self._collect_market_data()
                wallet_balances = await self._get_wallet_balances()
                gas_price = await self._get_gas_price()
            else:
                # Mock data for testing
                market_data = self._get_mock_market_data()
                wallet_balances = {"eth": 1.0, "usdc": 1000.0}
                gas_price = 25.0
            
            # Update state
            state["market_data"] = market_data
            state["wallet_balance"] = wallet_balances
            state["gas_price"] = gas_price
            state["network_congestion"] = self._calculate_congestion(gas_price)
            
            logger.info(f"Observation completed - {len(market_data)} pools observed")
            
        except Exception as e:
            logger.error(f"Observation failed: {e}")
            state["errors"].append({
                "node": "observation",
                "error": str(e),
                "timestamp": datetime.now()
            })
        
        return state
    
    async def _collect_market_data(self) -> Dict[str, Any]:
        """Collect real market data from CDP"""
        # This would use CDP manager to collect pool data
        return {}
    
    async def _get_wallet_balances(self) -> Dict[str, float]:
        """Get current wallet balances"""
        if self.cdp_manager:
            return await self.cdp_manager.get_balances(["eth", "usdc", "aero"])
        return {}
    
    async def _get_gas_price(self) -> float:
        """Get current gas price"""
        if self.cdp_manager:
            return await self.cdp_manager.get_gas_price()
        return 25.0
    
    def _get_mock_market_data(self) -> Dict[str, Any]:
        """Get mock market data for testing"""
        return {
            "pools": [
                {
                    "address": "0x123...",
                    "token0": "WETH",
                    "token1": "USDC",
                    "tvl": 1000000,
                    "apr": 12.5,
                    "volume_24h": 50000
                }
            ],
            "timestamp": datetime.now()
        }
    
    def _calculate_congestion(self, gas_price: float) -> float:
        """Calculate network congestion score from gas price"""
        # Simple congestion calculation based on gas price
        base_gas = 20.0
        if gas_price <= base_gas:
            return 0.0
        elif gas_price <= base_gas * 2:
            return 0.5
        else:
            return 1.0