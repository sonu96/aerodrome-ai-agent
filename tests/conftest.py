"""
Pytest configuration and shared fixtures for Aerodrome AI Agent tests.

This file provides shared fixtures, configuration, and test utilities
used across all test modules.
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

import pytest
from faker import Faker
from freezegun import freeze_time

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from aerodrome_ai_agent.brain.config import BrainConfig
from aerodrome_ai_agent.brain.state import create_initial_state, BrainState
from aerodrome_ai_agent.memory.config import MemoryConfig
from aerodrome_ai_agent.cdp.config import CDPConfig

# Initialize faker for generating test data
fake = Faker()


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async testing."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Configuration Fixtures
@pytest.fixture
def brain_config():
    """Provide test configuration for brain module."""
    return BrainConfig(
        confidence_threshold=0.7,
        risk_threshold=0.3,
        observation_interval=60,
        execution_timeout=30,
        max_position_size=0.2,
        max_slippage=0.02,
        min_trade_amount=10.0,
        max_trade_amount=1000.0,
        max_retries=3
    )


@pytest.fixture
def memory_config():
    """Provide test configuration for memory system."""
    return MemoryConfig(
        user_id="test_agent",
        vector_store="qdrant",
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="test_memories",
        max_memories=1000,
        max_memory_age_days=30,
        pattern_threshold=3,
        pruning_enabled=True,
        hot_tier_days=1,
        warm_tier_days=7,
        cold_tier_days=14
    )


@pytest.fixture
def cdp_config():
    """Provide test configuration for CDP integration."""
    return CDPConfig(
        api_key_name="test_key",
        api_key_private_key="test_private_key",
        network_id="base-sepolia",  # Use testnet
        wallet_id="test_wallet",
        wallet_seed="test_seed",
        requests_per_second=5,
        transaction_timeout=300,
        min_confirmations=1
    )


# State Fixtures
@pytest.fixture
def initial_brain_state():
    """Provide initial brain state for testing."""
    return create_initial_state()


@pytest.fixture
def sample_brain_state():
    """Provide a sample brain state with test data."""
    state = create_initial_state()
    state.update({
        "cycle_count": 5,
        "market_data": {
            "eth_price": 2500.0,
            "base_eth_price": 2498.5,
            "volatility": 0.15,
            "gas_price": 20.0
        },
        "wallet_balance": {
            "eth": 5.0,
            "usdc": 10000.0,
            "aerodrome_tokens": 1000.0
        },
        "opportunities": [
            {
                "type": "SWAP",
                "pool_address": "0x1234567890abcdef",
                "token_in": "USDC",
                "token_out": "ETH",
                "amount_in": 1000.0,
                "expected_output": 0.4,
                "score": 0.85,
                "confidence": 0.8
            }
        ],
        "selected_action": {
            "type": "SWAP",
            "pool_address": "0x1234567890abcdef",
            "token_in": "USDC",
            "token_out": "ETH",
            "amount_in": 1000.0,
            "expected_output": 0.4
        },
        "confidence_score": 0.85
    })
    return state


# Mock Data Fixtures
@pytest.fixture
def mock_pool_data():
    """Provide mock pool data for testing."""
    return {
        "address": "0x1234567890abcdef",
        "token0": {
            "address": "0xA0b86a33E6441c8C4C0a27b2B7fE9A7b6e3a1234",
            "symbol": "USDC",
            "decimals": 6,
            "balance": "1000000000000"
        },
        "token1": {
            "address": "0xB1c87a44F6552c8D4D1b38c3C7fE9B8c7f4b5678",
            "symbol": "WETH",
            "decimals": 18,
            "balance": "500000000000000000000"
        },
        "fee": 3000,
        "tvl": 50000000.0,
        "volume_24h": 2500000.0,
        "fee_growth": 0.01,
        "price": 2500.0,
        "liquidity": "75000000000000000000000"
    }


@pytest.fixture
def mock_transaction_data():
    """Provide mock transaction data for testing."""
    return {
        "hash": "0xabcdef1234567890",
        "from": "0x742d35Cc6Af0532f87b8C2d7aF40EC43d94f78d9",
        "to": "0x1234567890abcdef",
        "value": "1000000000000000000",
        "gas_limit": 150000,
        "gas_price": "20000000000",
        "gas_used": 142350,
        "status": "success",
        "block_number": 12345678,
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture
def mock_memory_data():
    """Provide mock memory data for testing."""
    base_time = datetime.now() - timedelta(hours=1)
    memories = []
    
    for i in range(10):
        memory = {
            "id": f"memory_{i}",
            "type": "experience",
            "timestamp": (base_time + timedelta(minutes=i*5)).isoformat(),
            "experience": {
                "action_type": fake.random_element(["SWAP", "ADD_LIQUIDITY", "REMOVE_LIQUIDITY"]),
                "pool": fake.ethereum_address(),
                "amount": fake.pyfloat(positive=True, max_value=1000),
                "confidence": fake.pyfloat(min_value=0.5, max_value=1.0)
            },
            "outcome": {
                "success": fake.boolean(chance_of_getting_true=80),
                "profit": fake.pyfloat(min_value=-100, max_value=100),
                "confidence": fake.pyfloat(min_value=0.5, max_value=1.0)
            }
        }
        memories.append(memory)
    
    return memories


# Environment Setup Fixtures
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env = {
        "TESTING": "true",
        "LOG_LEVEL": "DEBUG",
        "OPENAI_API_KEY": "test_openai_key",
        "CDP_API_KEY": "test_cdp_key",
        "CDP_PRIVATE_KEY": "test_cdp_private_key",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
    }
    
    with patch.dict(os.environ, test_env):
        yield


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = MagicMock(return_value=MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    ))
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="Test response"),
            finish_reason="stop"
        )]
    ))
    return mock_client


# CDP SDK Mocks
@pytest.fixture
def mock_cdp_wallet():
    """Mock CDP wallet for testing."""
    mock_wallet = AsyncMock()
    mock_wallet.id = "test_wallet_123"
    mock_wallet.network_id = "base-sepolia"
    mock_wallet.default_address = MagicMock()
    mock_wallet.default_address.address_id = "0x742d35Cc6Af0532f87b8C2d7aF40EC43d94f78d9"
    
    # Mock balance method
    async def mock_balance(asset):
        balances = {
            "eth": MagicMock(amount=5.0),
            "usdc": MagicMock(amount=10000.0)
        }
        return balances.get(asset, MagicMock(amount=0.0))
    
    mock_wallet.balance = mock_balance
    mock_wallet.addresses = [mock_wallet.default_address]
    
    return mock_wallet


@pytest.fixture
def mock_cdp_manager():
    """Mock CDP manager for testing."""
    from tests.mocks.cdp_mocks import MockCDPManager
    return MockCDPManager()


# Memory System Mocks
@pytest.fixture
def mock_memory_system():
    """Mock memory system for testing."""
    from tests.mocks.memory_mocks import MockMemorySystem
    return MockMemorySystem()


# Utility Functions
def assert_state_valid(state: BrainState):
    """Assert that a brain state has required fields."""
    required_fields = [
        "timestamp", "cycle_count", "execution_status", 
        "market_data", "wallet_balance", "opportunities"
    ]
    for field in required_fields:
        assert field in state, f"State missing required field: {field}"


def create_mock_opportunity(
    action_type: str = "SWAP",
    score: float = 0.8,
    confidence: float = 0.75
) -> Dict[str, Any]:
    """Create a mock opportunity for testing."""
    return {
        "type": action_type,
        "pool_address": fake.ethereum_address(),
        "token_in": fake.random_element(["USDC", "ETH", "WBTC"]),
        "token_out": fake.random_element(["USDC", "ETH", "WBTC"]),
        "amount_in": fake.pyfloat(positive=True, max_value=1000),
        "expected_output": fake.pyfloat(positive=True, max_value=10),
        "score": score,
        "confidence": confidence,
        "gas_estimate": fake.pyint(min_value=100000, max_value=200000),
        "slippage": fake.pyfloat(min_value=0.001, max_value=0.02)
    }


# Test Markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )
    config.addinivalue_line(
        "markers", "mock_only: mark test to run only with mocks"
    )


# Test Collection Hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration/ directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)