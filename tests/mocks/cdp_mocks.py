"""
Mock implementations for CDP SDK components.

Provides mock objects that simulate CDP SDK behavior without
requiring actual blockchain connections or API keys.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

from tests.fixtures.contract_data import (
    SUCCESSFUL_SIMULATION, FAILED_SIMULATION, SAMPLE_POOL_RESPONSE,
    SUCCESSFUL_SWAP_TX, FAILED_SWAP_TX, WALLET_RESPONSES,
    get_contract_address, create_transaction_receipt
)


class MockCDPWallet:
    """Mock CDP wallet for testing."""
    
    def __init__(self, wallet_id: str = "test_wallet", network_id: str = "base-sepolia"):
        self.id = wallet_id
        self.network_id = network_id
        self.default_address = MockAddress("0x742d35Cc6Af0532f87b8C2d7aF40EC43d94f78d9")
        self.addresses = [self.default_address]
        self._balances = {
            "eth": 5.0,
            "usdc": 10000.0,
            "usdt": 5000.0,
            "wbtc": 0.5
        }
    
    async def balance(self, asset: str):
        """Mock balance retrieval."""
        await asyncio.sleep(0.01)  # Simulate API delay
        amount = self._balances.get(asset.lower(), 0.0)
        return MockBalance(amount, asset.upper())
    
    def set_balance(self, asset: str, amount: float):
        """Set balance for testing."""
        self._balances[asset.lower()] = amount
    
    async def invoke_contract(
        self, 
        contract_address: str, 
        method: str, 
        abi: List[Dict],
        args: Dict[str, Any] = None,
        amount: float = None
    ):
        """Mock contract invocation."""
        await asyncio.sleep(0.1)  # Simulate transaction time
        
        # Simulate different outcomes based on method
        if method == "swapExactTokensForTokens":
            if args and args.get("amountIn", 0) > self._balances.get("usdc", 0):
                raise Exception("Insufficient balance")
            
            return MockInvocation(
                transaction_hash="0x1234567890abcdef",
                status="pending"
            )
        
        elif method == "addLiquidity":
            return MockInvocation(
                transaction_hash="0x2345678901bcdef0", 
                status="pending"
            )
        
        return MockInvocation(
            transaction_hash="0xabcdef1234567890",
            status="pending"
        )


class MockAddress:
    """Mock CDP address."""
    
    def __init__(self, address: str):
        self.address_id = address
    
    def __str__(self):
        return self.address_id


class MockBalance:
    """Mock CDP balance."""
    
    def __init__(self, amount: float, currency: str):
        self.amount = amount
        self.currency = currency
    
    def __float__(self):
        return self.amount


class MockInvocation:
    """Mock CDP contract invocation."""
    
    def __init__(self, transaction_hash: str, status: str = "pending"):
        self.transaction_hash = transaction_hash
        self.status = status
        self._wait_called = False
    
    async def wait(self, timeout: int = 300):
        """Mock wait for transaction confirmation."""
        await asyncio.sleep(0.1)  # Simulate confirmation time
        self._wait_called = True
        self.status = "confirmed"
        return MockTransactionReceipt(self.transaction_hash, True)


class MockTransactionReceipt:
    """Mock transaction receipt."""
    
    def __init__(self, tx_hash: str, success: bool = True):
        self.transaction_hash = tx_hash
        self.success = success
        self.gas_used = 150000 if success else 0
        self.block_number = 12345678
        self.timestamp = datetime.now()
        
        if success:
            self.logs = [
                {
                    "address": get_contract_address("usdc"),
                    "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"],
                    "data": "0x00000000000000000000000000000000000000000000000000000000000f4240"
                }
            ]
        else:
            self.logs = []
            self.error = "execution reverted"


class MockCDPManager:
    """Mock CDP Manager for testing."""
    
    def __init__(self, config=None):
        self.config = config
        self._wallet = None
        self._should_fail = False
        self._failure_reason = "NetworkError"
        self._simulation_results = {}
        self._pool_data = {}
    
    async def initialize_wallet(self) -> Dict[str, Any]:
        """Mock wallet initialization."""
        self._wallet = MockCDPWallet()
        return {
            "wallet_id": self._wallet.id,
            "network": self._wallet.network_id,
            "default_address": str(self._wallet.default_address),
            "balance_eth": 5.0,
            "addresses": [str(addr) for addr in self._wallet.addresses]
        }
    
    async def get_wallet_info(self) -> Dict[str, Any]:
        """Mock wallet info retrieval."""
        if not self._wallet:
            raise RuntimeError("Wallet not initialized")
        
        return {
            "wallet_id": self._wallet.id,
            "network": self._wallet.network_id,
            "default_address": str(self._wallet.default_address),
            "balance_eth": 5.0,
            "addresses": [str(addr) for addr in self._wallet.addresses]
        }
    
    async def get_balances(self, tokens: List[str] = None) -> Dict[str, float]:
        """Mock balance retrieval."""
        if not self._wallet:
            raise RuntimeError("Wallet not initialized")
        
        balances = {"eth": 5.0, "usdc": 10000.0, "usdt": 5000.0}
        
        if tokens:
            return {token: balances.get(token.lower(), 0.0) for token in tokens}
        
        return balances
    
    async def simulate_transaction(
        self,
        action: Dict[str, Any],
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Mock transaction simulation."""
        await asyncio.sleep(0.05)  # Simulate API call
        
        if self._should_fail:
            return {
                **FAILED_SIMULATION,
                "error": self._failure_reason
            }
        
        # Check for preset simulation result
        action_key = f"{action.get('type', 'UNKNOWN')}_{action.get('pool_address', 'default')}"
        if action_key in self._simulation_results:
            return self._simulation_results[action_key]
        
        # Default successful simulation
        simulation = SUCCESSFUL_SIMULATION.copy()
        simulation["timestamp"] = datetime.now()
        
        # Customize based on action type
        if action.get("type") == "SWAP":
            simulation.update({
                "expected_output": action.get("expected_output", 0.398),
                "price_impact": min(0.01, action.get("amount_in", 1000) / 100000)
            })
        
        return simulation
    
    async def invoke_contract(
        self,
        contract_address: str,
        method: str,
        abi: List[Dict],
        args: Dict[str, Any] = None,
        value: float = 0.0
    ) -> Dict[str, Any]:
        """Mock contract invocation."""
        if not self._wallet:
            raise RuntimeError("Wallet not initialized")
        
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        invocation = await self._wallet.invoke_contract(
            contract_address, method, abi, args, value
        )
        
        # Wait for confirmation
        receipt = await invocation.wait()
        
        return {
            "transaction_hash": invocation.transaction_hash,
            "status": "confirmed",
            "gas_used": receipt.gas_used,
            "block_number": receipt.block_number,
            "success": receipt.success
        }
    
    async def read_contract(
        self,
        contract_address: str,
        method: str,
        abi: List[Dict],
        args: List[Any] = None
    ) -> Any:
        """Mock contract reading."""
        await asyncio.sleep(0.02)  # Simulate API call
        
        # Return mock data based on method
        if method == "getReserves":
            return ["1000000000000", "400000000000000000000", int(datetime.now().timestamp())]
        elif method == "getAmountOut":
            amount_in = args[0] if args else 1000
            return str(int(amount_in * 0.398 * 1e18))  # Mock 0.398 ETH per 1000 USDC
        elif method == "balanceOf":
            return "10000000000"  # 10,000 tokens
        elif method == "totalSupply":
            return "100000000000000000000000000"  # 100M tokens
        
        return "0"
    
    async def get_pool_data(self, pool_address: str) -> Dict[str, Any]:
        """Mock pool data retrieval."""
        await asyncio.sleep(0.03)  # Simulate API call
        
        # Check for preset pool data
        if pool_address in self._pool_data:
            return self._pool_data[pool_address]
        
        # Return default pool data
        pool_data = SAMPLE_POOL_RESPONSE.copy()
        pool_data["address"] = pool_address
        return pool_data
    
    async def get_top_pools(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Mock top pools retrieval."""
        await asyncio.sleep(0.05)  # Simulate API call
        
        pools = []
        for i in range(min(limit, 10)):  # Return max 10 mock pools
            pool = SAMPLE_POOL_RESPONSE.copy()
            pool["address"] = f"0x{'0' * (40 - len(str(i)))}{i}"
            pool["tvl"] = 5000000 - (i * 500000)  # Decreasing TVL
            pool["volume_24h"] = 1000000 - (i * 100000)
            pools.append(pool)
        
        return pools
    
    async def get_gas_price(self) -> float:
        """Mock gas price retrieval."""
        await asyncio.sleep(0.01)
        return 20.0  # 20 gwei
    
    async def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Mock transaction status retrieval."""
        await asyncio.sleep(0.02)
        
        return {
            "hash": tx_hash,
            "status": "confirmed",
            "block_number": 12345678,
            "gas_used": 150000,
            "timestamp": datetime.now(),
            "success": not self._should_fail
        }
    
    async def wait_for_confirmation(
        self, 
        tx_hash: str, 
        confirmations: int = None
    ) -> Dict[str, Any]:
        """Mock transaction confirmation waiting."""
        await asyncio.sleep(0.1)  # Simulate confirmation time
        return await self.get_transaction_status(tx_hash)
    
    # Test utilities
    def set_should_fail(self, should_fail: bool, reason: str = "NetworkError"):
        """Configure mock to simulate failures."""
        self._should_fail = should_fail
        self._failure_reason = reason
    
    def set_simulation_result(self, action_type: str, pool_address: str, result: Dict):
        """Set custom simulation result for testing."""
        key = f"{action_type}_{pool_address}"
        self._simulation_results[key] = result
    
    def set_pool_data(self, pool_address: str, data: Dict):
        """Set custom pool data for testing."""
        self._pool_data[pool_address] = data
    
    def set_wallet_balance(self, asset: str, amount: float):
        """Set wallet balance for testing."""
        if self._wallet:
            self._wallet.set_balance(asset, amount)
    
    async def close(self):
        """Mock cleanup."""
        pass


class MockWalletManager:
    """Mock wallet manager for testing."""
    
    def __init__(self, config=None):
        self.config = config
    
    async def create_wallet(self) -> MockCDPWallet:
        """Mock wallet creation."""
        await asyncio.sleep(0.1)
        return MockCDPWallet()
    
    async def load_wallet(self, wallet_id: str, seed: str = None) -> MockCDPWallet:
        """Mock wallet loading."""
        await asyncio.sleep(0.05)
        return MockCDPWallet(wallet_id)


class MockContractManager:
    """Mock contract manager for testing."""
    
    def __init__(self, config=None):
        self.config = config
    
    async def read_contract(
        self, 
        address: str, 
        method: str, 
        abi: List[Dict], 
        args: List[Any] = None
    ) -> Any:
        """Mock contract reading."""
        await asyncio.sleep(0.02)
        return "mock_result"
    
    async def invoke_contract(
        self,
        wallet,
        address: str,
        method: str,
        abi: List[Dict],
        args: Dict[str, Any] = None,
        value: float = 0.0
    ) -> Dict[str, Any]:
        """Mock contract invocation."""
        await asyncio.sleep(0.1)
        return {
            "transaction_hash": "0xmockhash",
            "status": "confirmed",
            "gas_used": 150000
        }
    
    async def get_pool_data(self, pool_address: str) -> Dict[str, Any]:
        """Mock pool data retrieval."""
        pool_data = SAMPLE_POOL_RESPONSE.copy()
        pool_data["address"] = pool_address
        return pool_data
    
    async def get_top_pools(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Mock top pools retrieval.""" 
        return [SAMPLE_POOL_RESPONSE.copy() for _ in range(min(limit, 5))]