"""
Tests for the CDP SDK integration.

Comprehensive test suite covering wallet management, contract interactions,
transaction handling, and blockchain operations.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from aerodrome_ai_agent.cdp.manager import CDPManager
from aerodrome_ai_agent.cdp.config import CDPConfig
from tests.mocks.cdp_mocks import (
    MockCDPManager, MockCDPWallet, MockWalletManager, MockContractManager
)
from tests.fixtures.contract_data import (
    BASE_CONTRACTS, SUCCESSFUL_SWAP_TX, FAILED_SWAP_TX,
    SUCCESSFUL_SIMULATION, FAILED_SIMULATION, SAMPLE_POOL_RESPONSE,
    get_contract_abi, create_transaction_receipt, get_error_response
)


class TestCDPConfig:
    """Test CDP configuration handling."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CDPConfig()
        
        assert config.network_id == "base-mainnet"
        assert config.requests_per_second == 10
        assert config.transaction_timeout == 600
        assert config.min_confirmations == 2
        assert config.max_gas_price == 100.0
    
    def test_config_validation_valid(self):
        """Test valid configuration passes validation."""
        config = CDPConfig(
            api_key_name="test_key",
            api_key_private_key="test_private_key",
            network_id="base-sepolia",
            requests_per_second=5
        )
        
        # Should not raise
        config.validate()
    
    def test_config_validation_missing_api_key(self):
        """Test validation fails with missing API key."""
        config = CDPConfig(api_key_name="")
        
        with pytest.raises(ValueError, match="API key name is required"):
            config.validate()
    
    def test_config_validation_missing_private_key(self):
        """Test validation fails with missing private key."""
        config = CDPConfig(api_key_private_key="")
        
        with pytest.raises(ValueError, match="API private key is required"):
            config.validate()
    
    def test_config_validation_invalid_network(self):
        """Test validation fails with invalid network."""
        config = CDPConfig(network_id="invalid-network")
        
        with pytest.raises(ValueError, match="Invalid network_id"):
            config.validate()
    
    def test_config_validation_invalid_requests_per_second(self):
        """Test validation fails with invalid request rate."""
        config = CDPConfig(requests_per_second=0)
        
        with pytest.raises(ValueError, match="requests_per_second must be positive"):
            config.validate()
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = CDPConfig(
            api_key_name="test_key",
            network_id="base-sepolia",
            requests_per_second=5
        )
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["api_key_name"] == "test_key"
        assert config_dict["network_id"] == "base-sepolia"
        assert config_dict["requests_per_second"] == 5
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        env_vars = {
            "CDP_API_KEY": "env_test_key",
            "CDP_PRIVATE_KEY": "env_private_key",
            "CDP_NETWORK_ID": "base-sepolia",
            "CDP_WALLET_ID": "env_wallet_123"
        }
        
        with patch.dict('os.environ', env_vars):
            config = CDPConfig.from_env()
            
            assert config.api_key_name == "env_test_key"
            assert config.api_key_private_key == "env_private_key"
            assert config.network_id == "base-sepolia"
            assert config.wallet_id == "env_wallet_123"


class TestCDPManager:
    """Test main CDP manager functionality."""
    
    @pytest.fixture
    def cdp_manager(self, cdp_config):
        """Create CDP manager for testing."""
        return CDPManager(cdp_config)
    
    @pytest.fixture
    def mock_cdp_manager(self, cdp_config):
        """Create mock CDP manager for testing."""
        return MockCDPManager(cdp_config)
    
    def test_cdp_manager_initialization(self, cdp_config):
        """Test CDP manager initialization."""
        with patch('aerodrome_ai_agent.cdp.manager.Cdp') as mock_cdp:
            manager = CDPManager(cdp_config)
            
            assert manager.config == cdp_config
            mock_cdp.configure.assert_called_once()
    
    def test_cdp_manager_initialization_without_sdk(self, cdp_config):
        """Test CDP manager initialization without SDK."""
        with patch('aerodrome_ai_agent.cdp.manager.Cdp', None):
            manager = CDPManager(cdp_config)
            
            assert manager.config == cdp_config
    
    @pytest.mark.asyncio
    async def test_initialize_wallet_new(self, mock_cdp_manager):
        """Test initializing a new wallet.""" 
        # Configure no existing wallet
        mock_cdp_manager.config.wallet_id = None
        
        wallet_info = await mock_cdp_manager.initialize_wallet()
        
        assert "wallet_id" in wallet_info
        assert "network" in wallet_info
        assert "default_address" in wallet_info
        assert "balance_eth" in wallet_info
        assert wallet_info["balance_eth"] == 5.0
    
    @pytest.mark.asyncio
    async def test_initialize_wallet_existing(self, mock_cdp_manager):
        """Test loading an existing wallet."""
        # Configure existing wallet
        mock_cdp_manager.config.wallet_id = "existing_wallet_123"
        
        wallet_info = await mock_cdp_manager.initialize_wallet()
        
        assert wallet_info["wallet_id"] == "existing_wallet_123"
        assert "default_address" in wallet_info
    
    @pytest.mark.asyncio
    async def test_initialize_wallet_failure(self, mock_cdp_manager):
        """Test wallet initialization failure."""
        mock_cdp_manager.set_should_fail(True, "WalletInitializationError")
        
        with pytest.raises(Exception, match="WalletInitializationError"):
            await mock_cdp_manager.initialize_wallet()
    
    @pytest.mark.asyncio
    async def test_get_wallet_info_success(self, mock_cdp_manager):
        """Test successful wallet info retrieval."""
        await mock_cdp_manager.initialize_wallet()
        
        wallet_info = await mock_cdp_manager.get_wallet_info()
        
        assert "wallet_id" in wallet_info
        assert "network" in wallet_info
        assert "default_address" in wallet_info
        assert "balance_eth" in wallet_info
        assert "addresses" in wallet_info
    
    @pytest.mark.asyncio
    async def test_get_wallet_info_not_initialized(self, mock_cdp_manager):
        """Test wallet info retrieval without initialization."""
        with pytest.raises(RuntimeError, match="Wallet not initialized"):
            await mock_cdp_manager.get_wallet_info()
    
    @pytest.mark.asyncio
    async def test_get_balances_success(self, mock_cdp_manager):
        """Test successful balance retrieval."""
        await mock_cdp_manager.initialize_wallet()
        
        balances = await mock_cdp_manager.get_balances(["eth", "usdc", "usdt"])
        
        assert "eth" in balances
        assert "usdc" in balances
        assert "usdt" in balances
        assert balances["eth"] == 5.0
        assert balances["usdc"] == 10000.0
    
    @pytest.mark.asyncio
    async def test_get_balances_all_tokens(self, mock_cdp_manager):
        """Test balance retrieval for all tokens."""
        await mock_cdp_manager.initialize_wallet()
        
        balances = await mock_cdp_manager.get_balances()
        
        assert "eth" in balances
        assert "usdc" in balances
        assert "usdt" in balances
    
    @pytest.mark.asyncio
    async def test_get_balances_not_initialized(self, mock_cdp_manager):
        """Test balance retrieval without wallet initialization."""
        with pytest.raises(RuntimeError, match="Wallet not initialized"):
            await mock_cdp_manager.get_balances()


class TestContractInteractions:
    """Test smart contract interactions."""
    
    @pytest.fixture
    def initialized_manager(self, mock_cdp_manager):
        """Create initialized CDP manager.""" 
        async def setup():
            await mock_cdp_manager.initialize_wallet()
            return mock_cdp_manager
        return setup()
    
    @pytest.mark.asyncio
    async def test_read_contract_success(self, mock_cdp_manager):
        """Test successful contract reading."""
        pool_abi = get_contract_abi("aerodrome_pool")
        pool_address = "0x1234567890abcdef1234567890abcdef12345678"
        
        result = await mock_cdp_manager.read_contract(
            pool_address, "getReserves", pool_abi
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_read_contract_with_args(self, mock_cdp_manager):
        """Test contract reading with arguments."""
        pool_abi = get_contract_abi("aerodrome_pool")
        pool_address = "0x1234567890abcdef1234567890abcdef12345678"
        
        result = await mock_cdp_manager.read_contract(
            pool_address, "getAmountOut", pool_abi, [1000, BASE_CONTRACTS["usdc"]]
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_invoke_contract_success(self, mock_cdp_manager):
        """Test successful contract invocation."""
        await mock_cdp_manager.initialize_wallet()
        
        router_abi = get_contract_abi("aerodrome_router")
        router_address = BASE_CONTRACTS["aerodrome_router"]
        
        args = {
            "amountIn": 1000,
            "amountOutMin": 395,
            "routes": [{"from": BASE_CONTRACTS["usdc"], "to": BASE_CONTRACTS["weth"]}],
            "to": "0x742d35Cc6Af0532f87b8C2d7aF40EC43d94f78d9",
            "deadline": int(datetime.now().timestamp()) + 3600
        }
        
        result = await mock_cdp_manager.invoke_contract(
            router_address, "swapExactTokensForTokens", router_abi, args
        )
        
        assert result["status"] == "confirmed"
        assert "transaction_hash" in result
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_invoke_contract_failure(self, mock_cdp_manager):
        """Test contract invocation failure."""
        await mock_cdp_manager.initialize_wallet()
        mock_cdp_manager.set_should_fail(True, "InsufficientBalance")
        
        router_abi = get_contract_abi("aerodrome_router")
        router_address = BASE_CONTRACTS["aerodrome_router"]
        
        with pytest.raises(Exception, match="InsufficientBalance"):
            await mock_cdp_manager.invoke_contract(
                router_address, "swapExactTokensForTokens", router_abi, {}
            )
    
    @pytest.mark.asyncio
    async def test_invoke_contract_not_initialized(self, mock_cdp_manager):
        """Test contract invocation without wallet initialization."""
        router_abi = get_contract_abi("aerodrome_router")
        router_address = BASE_CONTRACTS["aerodrome_router"]
        
        with pytest.raises(RuntimeError, match="Wallet not initialized"):
            await mock_cdp_manager.invoke_contract(
                router_address, "swapExactTokensForTokens", router_abi, {}
            )


class TestTransactionSimulation:
    """Test transaction simulation functionality."""
    
    @pytest.fixture
    def swap_action(self):
        """Create swap action for testing."""
        return {
            "type": "SWAP",
            "pool_address": "0x1234567890abcdef1234567890abcdef12345678",
            "token_in": "USDC",
            "token_out": "WETH", 
            "amount_in": 1000.0,
            "expected_output": 0.398
        }
    
    @pytest.fixture
    def liquidity_action(self):
        """Create add liquidity action for testing."""
        return {
            "type": "ADD_LIQUIDITY",
            "pool_address": "0x2345678901bcdef02345678901bcdef023456789",
            "token_a": "USDC",
            "token_b": "USDT",
            "amount_a": 5000.0,
            "amount_b": 5000.0,
            "expected_lp_tokens": 10000.0
        }
    
    @pytest.mark.asyncio
    async def test_simulate_swap_success(self, mock_cdp_manager, swap_action):
        """Test successful swap simulation."""
        # Configure successful simulation
        mock_cdp_manager.set_simulation_result(
            "SWAP",
            swap_action["pool_address"],
            {
                "success": True,
                "profitable": True,
                "gas_estimate": 125000,
                "expected_output": 0.398,
                "slippage": 0.005
            }
        )
        
        result = await mock_cdp_manager.simulate_transaction(swap_action)
        
        assert result["success"] is True
        assert result["profitable"] is True
        assert result["gas_estimate"] == 125000
        assert result["expected_output"] == 0.398
    
    @pytest.mark.asyncio
    async def test_simulate_swap_failure(self, mock_cdp_manager, swap_action):
        """Test failed swap simulation."""
        mock_cdp_manager.set_should_fail(True, "InsufficientLiquidity")
        
        result = await mock_cdp_manager.simulate_transaction(swap_action)
        
        assert result["success"] is False
        assert "error" in result
        assert "InsufficientLiquidity" in result["error"]
    
    @pytest.mark.asyncio
    async def test_simulate_add_liquidity(self, mock_cdp_manager, liquidity_action):
        """Test add liquidity simulation."""
        result = await mock_cdp_manager.simulate_transaction(liquidity_action)
        
        assert result["success"] is True
        assert "lp_tokens" in result or "expected_lp_tokens" in result
    
    @pytest.mark.asyncio
    async def test_simulate_with_parameters(self, mock_cdp_manager, swap_action):
        """Test simulation with custom parameters."""
        params = {
            "slippage": 0.02,
            "deadline": int(datetime.now().timestamp()) + 1800
        }
        
        result = await mock_cdp_manager.simulate_transaction(swap_action, params)
        
        assert result["success"] is True
        assert "slippage" in result
    
    @pytest.mark.asyncio
    async def test_simulate_high_slippage(self, mock_cdp_manager, swap_action):
        """Test simulation with high slippage scenario.""" 
        # Configure high slippage simulation
        mock_cdp_manager.set_simulation_result(
            "SWAP",
            swap_action["pool_address"],
            {
                "success": True,
                "profitable": False,
                "price_impact": 0.05,  # 5% price impact
                "slippage": 0.03
            }
        )
        
        result = await mock_cdp_manager.simulate_transaction(swap_action)
        
        assert result["success"] is True
        assert result["profitable"] is False
        assert result["price_impact"] == 0.05


class TestTransactionExecution:
    """Test transaction execution and monitoring.""" 
    
    @pytest.mark.asyncio
    async def test_get_gas_price(self, mock_cdp_manager):
        """Test gas price retrieval."""
        gas_price = await mock_cdp_manager.get_gas_price()
        
        assert isinstance(gas_price, (int, float))
        assert gas_price > 0
    
    @pytest.mark.asyncio
    async def test_get_transaction_status_confirmed(self, mock_cdp_manager):
        """Test transaction status for confirmed transaction."""
        tx_hash = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        
        status = await mock_cdp_manager.get_transaction_status(tx_hash)
        
        assert status["hash"] == tx_hash
        assert status["status"] == "confirmed"
        assert "gas_used" in status
        assert "block_number" in status
    
    @pytest.mark.asyncio
    async def test_get_transaction_status_failed(self, mock_cdp_manager):
        """Test transaction status for failed transaction."""
        mock_cdp_manager.set_should_fail(True, "TransactionFailed")
        tx_hash = "0x9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba"
        
        status = await mock_cdp_manager.get_transaction_status(tx_hash)
        
        assert status["hash"] == tx_hash
        assert status["success"] is False
    
    @pytest.mark.asyncio
    async def test_wait_for_confirmation_success(self, mock_cdp_manager):
        """Test waiting for transaction confirmation."""
        tx_hash = "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        
        result = await mock_cdp_manager.wait_for_confirmation(tx_hash, confirmations=1)
        
        assert result["hash"] == tx_hash
        assert result["status"] == "confirmed"
    
    @pytest.mark.asyncio
    async def test_wait_for_confirmation_timeout(self, mock_cdp_manager):
        """Test transaction confirmation timeout."""
        # Set very short timeout for testing
        mock_cdp_manager.config.transaction_timeout = 1
        tx_hash = "0xtimeout1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        
        # Mock status to always return pending
        with patch.object(mock_cdp_manager, 'get_transaction_status') as mock_status:
            mock_status.return_value = {
                "hash": tx_hash,
                "status": "pending"
            }
            
            result = await mock_cdp_manager.wait_for_confirmation(tx_hash)
            
            assert result["status"] == "timeout"
            assert "error" in result


class TestPoolDataRetrieval:
    """Test pool data and market information retrieval."""
    
    @pytest.mark.asyncio
    async def test_get_pool_data_success(self, mock_cdp_manager):
        """Test successful pool data retrieval."""
        pool_address = "0x1234567890abcdef1234567890abcdef12345678"
        
        # Set custom pool data
        custom_pool_data = {
            **SAMPLE_POOL_RESPONSE,
            "address": pool_address,
            "tvl": 2500000.0,
            "volume_24h": 500000.0
        }
        mock_cdp_manager.set_pool_data(pool_address, custom_pool_data)
        
        pool_data = await mock_cdp_manager.get_pool_data(pool_address)
        
        assert pool_data["address"] == pool_address
        assert "token0" in pool_data
        assert "token1" in pool_data
        assert "tvl" in pool_data
        assert "volume_24h" in pool_data
        assert pool_data["tvl"] == 2500000.0
    
    @pytest.mark.asyncio
    async def test_get_pool_data_default(self, mock_cdp_manager):
        """Test pool data retrieval with default data."""
        pool_address = "0x9999999999999999999999999999999999999999"
        
        pool_data = await mock_cdp_manager.get_pool_data(pool_address)
        
        assert pool_data["address"] == pool_address
        assert "token0" in pool_data
        assert "token1" in pool_data
        assert "fee" in pool_data
    
    @pytest.mark.asyncio
    async def test_get_top_pools(self, mock_cdp_manager):
        """Test top pools retrieval."""
        top_pools = await mock_cdp_manager.get_top_pools(limit=5)
        
        assert isinstance(top_pools, list)
        assert len(top_pools) <= 5
        
        if top_pools:
            # Verify pool structure
            pool = top_pools[0]
            assert "address" in pool
            assert "tvl" in pool
            assert "volume_24h" in pool
            assert "token0" in pool
            assert "token1" in pool
            
            # Verify pools are sorted by TVL (descending)
            if len(top_pools) > 1:
                assert top_pools[0]["tvl"] >= top_pools[1]["tvl"]
    
    @pytest.mark.asyncio
    async def test_get_top_pools_large_limit(self, mock_cdp_manager):
        """Test top pools retrieval with large limit."""
        top_pools = await mock_cdp_manager.get_top_pools(limit=50)
        
        assert isinstance(top_pools, list)
        assert len(top_pools) <= 50  # Mock returns max 10
    
    @pytest.mark.asyncio
    async def test_get_top_pools_error(self, mock_cdp_manager):
        """Test top pools retrieval with error."""
        mock_cdp_manager.set_should_fail(True, "NetworkError")
        
        with pytest.raises(Exception, match="NetworkError"):
            await mock_cdp_manager.get_top_pools()


class TestErrorHandling:
    """Test error handling in CDP operations."""
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, mock_cdp_manager):
        """Test handling of network errors."""
        mock_cdp_manager.set_should_fail(True, "NetworkTimeout")
        
        with pytest.raises(Exception, match="NetworkTimeout"):
            await mock_cdp_manager.initialize_wallet()
    
    @pytest.mark.asyncio
    async def test_insufficient_balance_error(self, mock_cdp_manager):
        """Test handling of insufficient balance errors."""
        await mock_cdp_manager.initialize_wallet()
        
        # Set very low balance
        mock_cdp_manager.set_wallet_balance("usdc", 1.0)
        mock_cdp_manager.set_should_fail(True, "InsufficientBalance")
        
        router_abi = get_contract_abi("aerodrome_router")
        router_address = BASE_CONTRACTS["aerodrome_router"]
        
        with pytest.raises(Exception, match="InsufficientBalance"):
            await mock_cdp_manager.invoke_contract(
                router_address, "swapExactTokensForTokens", router_abi,
                {"amountIn": 1000}  # More than balance
            )
    
    @pytest.mark.asyncio
    async def test_gas_price_error(self, mock_cdp_manager):
        """Test handling of gas price errors."""
        # Mock high gas price scenario
        with patch.object(mock_cdp_manager, 'get_gas_price', return_value=150.0):
            # This would exceed max gas price in config (typically 100 gwei)
            gas_price = await mock_cdp_manager.get_gas_price()
            assert gas_price == 150.0
    
    @pytest.mark.asyncio
    async def test_transaction_revert_error(self, mock_cdp_manager):
        """Test handling of transaction revert errors."""
        await mock_cdp_manager.initialize_wallet()
        mock_cdp_manager.set_should_fail(True, "execution reverted: EXCESSIVE_SLIPPAGE")
        
        router_abi = get_contract_abi("aerodrome_router")
        router_address = BASE_CONTRACTS["aerodrome_router"]
        
        with pytest.raises(Exception, match="EXCESSIVE_SLIPPAGE"):
            await mock_cdp_manager.invoke_contract(
                router_address, "swapExactTokensForTokens", router_abi, {}
            )
    
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self, mock_cdp_manager):
        """Test handling of rate limiting."""
        # Simulate multiple rapid requests
        requests = []
        for i in range(5):
            requests.append(mock_cdp_manager.get_gas_price())
        
        # All requests should complete (mock doesn't actually rate limit)
        results = await asyncio.gather(*requests, return_exceptions=True)
        
        # Verify no exceptions were raised
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, (int, float))


class TestCDPIntegration:
    """Integration tests for CDP components."""
    
    @pytest.mark.asyncio
    async def test_full_swap_workflow(self, mock_cdp_manager):
        """Test complete swap workflow from simulation to execution.""" 
        # Initialize wallet
        await mock_cdp_manager.initialize_wallet()
        
        # Define swap action
        swap_action = {
            "type": "SWAP",
            "pool_address": "0x1234567890abcdef1234567890abcdef12345678",
            "token_in": "USDC",
            "token_out": "WETH",
            "amount_in": 1000.0,
            "expected_output": 0.398
        }
        
        # Simulate transaction
        simulation = await mock_cdp_manager.simulate_transaction(swap_action)
        
        assert simulation["success"] is True
        assert simulation["profitable"] is True
        
        # Execute transaction
        router_abi = get_contract_abi("aerodrome_router")
        router_address = BASE_CONTRACTS["aerodrome_router"]
        
        args = {
            "amountIn": int(swap_action["amount_in"] * 1e6),  # USDC has 6 decimals
            "amountOutMin": int(swap_action["expected_output"] * 0.99 * 1e18),  # ETH has 18 decimals
            "routes": [{
                "from": BASE_CONTRACTS["usdc"],
                "to": BASE_CONTRACTS["weth"],
                "stable": False,
                "factory": BASE_CONTRACTS["aerodrome_factory"]
            }],
            "to": "0x742d35Cc6Af0532f87b8C2d7aF40EC43d94f78d9",
            "deadline": int(datetime.now().timestamp()) + 3600
        }
        
        result = await mock_cdp_manager.invoke_contract(
            router_address, "swapExactTokensForTokens", router_abi, args
        )
        
        assert result["status"] == "confirmed"
        assert result["success"] is True
        
        # Wait for confirmation
        confirmation = await mock_cdp_manager.wait_for_confirmation(
            result["transaction_hash"], confirmations=1
        )
        
        assert confirmation["status"] == "confirmed"
    
    @pytest.mark.asyncio
    async def test_full_liquidity_workflow(self, mock_cdp_manager):
        """Test complete add liquidity workflow."""
        # Initialize wallet
        await mock_cdp_manager.initialize_wallet()
        
        # Define liquidity action
        liquidity_action = {
            "type": "ADD_LIQUIDITY",
            "pool_address": "0x2345678901bcdef02345678901bcdef023456789",
            "token_a": "USDC",
            "token_b": "USDT",
            "amount_a": 5000.0,
            "amount_b": 5000.0
        }
        
        # Simulate transaction
        simulation = await mock_cdp_manager.simulate_transaction(liquidity_action)
        
        assert simulation["success"] is True
        
        # Execute transaction
        router_abi = get_contract_abi("aerodrome_router")
        router_address = BASE_CONTRACTS["aerodrome_router"]
        
        args = {
            "tokenA": BASE_CONTRACTS["usdc"],
            "tokenB": BASE_CONTRACTS["usdt"],
            "stable": True,  # USDC-USDT is a stable pair
            "amountADesired": int(liquidity_action["amount_a"] * 1e6),
            "amountBDesired": int(liquidity_action["amount_b"] * 1e6),
            "amountAMin": int(liquidity_action["amount_a"] * 0.99 * 1e6),
            "amountBMin": int(liquidity_action["amount_b"] * 0.99 * 1e6),
            "to": "0x742d35Cc6Af0532f87b8C2d7aF40EC43d94f78d9",
            "deadline": int(datetime.now().timestamp()) + 3600
        }
        
        result = await mock_cdp_manager.invoke_contract(
            router_address, "addLiquidity", router_abi, args
        )
        
        assert result["status"] == "confirmed"
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, mock_cdp_manager):
        """Test error recovery in CDP workflows."""
        # Initialize wallet
        await mock_cdp_manager.initialize_wallet()
        
        # First attempt fails
        mock_cdp_manager.set_should_fail(True, "TemporaryNetworkError")
        
        swap_action = {
            "type": "SWAP",
            "pool_address": "0x1234567890abcdef1234567890abcdef12345678",
            "token_in": "USDC",
            "token_out": "WETH",
            "amount_in": 1000.0
        }
        
        # Should fail
        result = await mock_cdp_manager.simulate_transaction(swap_action)
        assert result["success"] is False
        assert "TemporaryNetworkError" in result["error"]
        
        # Recover and retry
        mock_cdp_manager.set_should_fail(False)
        
        # Should succeed
        result = await mock_cdp_manager.simulate_transaction(swap_action)
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_wallet_balance_tracking(self, mock_cdp_manager):
        """Test wallet balance tracking throughout operations."""
        # Initialize wallet
        await mock_cdp_manager.initialize_wallet()
        
        # Get initial balances
        initial_balances = await mock_cdp_manager.get_balances(["eth", "usdc"])
        
        assert initial_balances["eth"] == 5.0
        assert initial_balances["usdc"] == 10000.0
        
        # Simulate balance changes after transaction
        mock_cdp_manager.set_wallet_balance("usdc", 9000.0)  # Spent 1000 USDC
        mock_cdp_manager.set_wallet_balance("eth", 5.398)    # Gained ~0.398 ETH
        
        # Get updated balances
        updated_balances = await mock_cdp_manager.get_balances(["eth", "usdc"])
        
        assert updated_balances["usdc"] == 9000.0
        assert updated_balances["eth"] == 5.398
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_cdp_manager):
        """Test handling concurrent CDP operations."""
        # Initialize wallet
        await mock_cdp_manager.initialize_wallet()
        
        # Start multiple concurrent operations
        operations = [
            mock_cdp_manager.get_gas_price(),
            mock_cdp_manager.get_pool_data("0x1234567890abcdef1234567890abcdef12345678"),
            mock_cdp_manager.get_top_pools(5),
            mock_cdp_manager.get_balances(["eth", "usdc"])
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*operations, return_exceptions=True)
        
        # Verify all operations completed successfully
        assert len(results) == 4
        for result in results:
            assert not isinstance(result, Exception)
        
        gas_price, pool_data, top_pools, balances = results
        
        assert isinstance(gas_price, (int, float))
        assert isinstance(pool_data, dict)
        assert isinstance(top_pools, list)
        assert isinstance(balances, dict)