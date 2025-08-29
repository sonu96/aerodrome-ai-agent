"""
Contract and blockchain data fixtures for testing CDP integration.

Contains sample contract ABIs, transaction data, and blockchain responses
for testing CDP SDK integration.
"""

from datetime import datetime
from typing import Dict, Any, List


# Sample Aerodrome contract ABIs
AERODROME_POOL_ABI = [
    {
        "inputs": [],
        "name": "token0",
        "outputs": [{"type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "token1", 
        "outputs": [{"type": "address"}],
        "stateMutability": "view", 
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"type": "uint112", "name": "_reserve0"},
            {"type": "uint112", "name": "_reserve1"},
            {"type": "uint32", "name": "_blockTimestampLast"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"type": "uint256", "name": "amountIn"},
            {"type": "address", "name": "tokenIn"}
        ],
        "name": "getAmountOut",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"type": "uint256", "name": "amount0Out"},
            {"type": "uint256", "name": "amount1Out"},
            {"type": "address", "name": "to"},
            {"type": "bytes", "name": "data"}
        ],
        "name": "swap",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

AERODROME_ROUTER_ABI = [
    {
        "inputs": [
            {"type": "uint256", "name": "amountIn"},
            {"type": "uint256", "name": "amountOutMin"},
            {
                "type": "tuple[]",
                "name": "routes",
                "components": [
                    {"type": "address", "name": "from"},
                    {"type": "address", "name": "to"},
                    {"type": "bool", "name": "stable"},
                    {"type": "address", "name": "factory"}
                ]
            },
            {"type": "address", "name": "to"},
            {"type": "uint256", "name": "deadline"}
        ],
        "name": "swapExactTokensForTokens",
        "outputs": [{"type": "uint256[]", "name": "amounts"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"type": "address", "name": "tokenA"},
            {"type": "address", "name": "tokenB"},
            {"type": "bool", "name": "stable"},
            {"type": "uint256", "name": "amountADesired"},
            {"type": "uint256", "name": "amountBDesired"},
            {"type": "uint256", "name": "amountAMin"},
            {"type": "uint256", "name": "amountBMin"},
            {"type": "address", "name": "to"},
            {"type": "uint256", "name": "deadline"}
        ],
        "name": "addLiquidity",
        "outputs": [
            {"type": "uint256", "name": "amountA"},
            {"type": "uint256", "name": "amountB"},
            {"type": "uint256", "name": "liquidity"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

ERC20_ABI = [
    {
        "inputs": [{"type": "address", "name": "owner"}],
        "name": "balanceOf",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"type": "address", "name": "spender"},
            {"type": "uint256", "name": "amount"}
        ],
        "name": "approve",
        "outputs": [{"type": "bool"}],
        "stateMutability": "nonpayable", 
        "type": "function"
    },
    {
        "inputs": [
            {"type": "address", "name": "to"},
            {"type": "uint256", "name": "amount"}
        ],
        "name": "transfer",
        "outputs": [{"type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

# Contract addresses for Base network
BASE_CONTRACTS = {
    "aerodrome_router": "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43",
    "aerodrome_factory": "0x420DD381b31aEf6683db6B902084cB0FFECe40Da",
    "usdc": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", 
    "weth": "0x4200000000000000000000000000000000000006",
    "usdt": "0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2",
    "wbtc": "0x2297aEbD383787A160DD0d9F71508148769342E3"
}

# Sample transaction data
SUCCESSFUL_SWAP_TX = {
    "hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    "from": "0x742d35Cc6Af0532f87b8C2d7aF40EC43d94f78d9",
    "to": "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43",
    "value": "0",
    "gas_limit": 150000,
    "gas_price": "20000000000",
    "gas_used": 142350,
    "status": "success",
    "block_number": 12345678,
    "block_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
    "timestamp": datetime.now().isoformat(),
    "logs": [
        {
            "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "topics": [
                "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                "0x000000000000000000000000742d35Cc6Af0532f87b8C2d7aF40EC43d94f78d9",
                "0x000000000000000000000000cF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
            ],
            "data": "0x00000000000000000000000000000000000000000000000000000000000f4240"
        }
    ]
}

FAILED_SWAP_TX = {
    "hash": "0x9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba", 
    "from": "0x742d35Cc6Af0532f87b8C2d7aF40EC43d94f78d9",
    "to": "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43",
    "value": "0",
    "gas_limit": 150000,
    "gas_price": "50000000000",
    "gas_used": 0,
    "status": "failed",
    "error": "execution reverted: Aerodrome: EXCESSIVE_SLIPPAGE",
    "block_number": 12345679,
    "timestamp": datetime.now().isoformat()
}

# Simulation responses
SUCCESSFUL_SIMULATION = {
    "success": True,
    "profitable": True,
    "gas_estimate": 142350,
    "gas_price": 20.0,
    "expected_output": 0.398,
    "price_impact": 0.002,
    "minimum_output": 0.394,
    "slippage": 0.01,
    "route": [
        {
            "from": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "to": "0x4200000000000000000000000000000000000006", 
            "stable": False,
            "pool": "0x1234567890abcdef1234567890abcdef12345678"
        }
    ],
    "timestamp": datetime.now()
}

FAILED_SIMULATION = {
    "success": False,
    "profitable": False,
    "error": "Insufficient liquidity for trade size",
    "gas_estimate": 0,
    "price_impact": 0.15,
    "timestamp": datetime.now()
}

# Pool data responses
SAMPLE_POOL_RESPONSE = {
    "address": "0x1234567890abcdef1234567890abcdef12345678",
    "token0": {
        "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "symbol": "USDC",
        "name": "USD Coin",
        "decimals": 6,
        "balance": "1000000000000"  # 1M USDC
    },
    "token1": {
        "address": "0x4200000000000000000000000000000000000006",
        "symbol": "WETH", 
        "name": "Wrapped Ether",
        "decimals": 18,
        "balance": "400000000000000000000"  # 400 ETH
    },
    "fee": 3000,  # 0.3%
    "stable": False,
    "reserves": {
        "reserve0": "1000000000000",
        "reserve1": "400000000000000000000",
        "timestamp": int(datetime.now().timestamp())
    },
    "tvl": 2500000.0,
    "volume_24h": 500000.0,
    "fees_24h": 1500.0,
    "price": 2500.0,
    "liquidity": "75000000000000000000000"
}

# Error responses
CONTRACT_ERROR_RESPONSES = {
    "revert_with_reason": {
        "error": "ContractExecutionError",
        "message": "execution reverted: Aerodrome: EXCESSIVE_SLIPPAGE",
        "code": -32000,
        "data": "0x08c379a0000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000234165726f64726f6d653a2045584345535349564520534c495050414745000000"
    },
    "out_of_gas": {
        "error": "OutOfGasError",
        "message": "out of gas",
        "code": -32000
    },
    "insufficient_funds": {
        "error": "InsufficientFundsError", 
        "message": "insufficient funds for gas * price + value",
        "code": -32000
    },
    "nonce_too_low": {
        "error": "NonceError",
        "message": "nonce too low",
        "code": -32000
    }
}

# Gas price data
GAS_PRICE_DATA = {
    "slow": {"price": 10.0, "wait_time": 600},
    "standard": {"price": 20.0, "wait_time": 300},
    "fast": {"price": 30.0, "wait_time": 60},
    "rapid": {"price": 50.0, "wait_time": 15}
}

# Network status
NETWORK_STATUS = {
    "healthy": {
        "status": "healthy",
        "block_number": 12345678,
        "block_time": 2.1,
        "pending_transactions": 150,
        "network_hashrate": "500 TH/s"
    },
    "congested": {
        "status": "congested", 
        "block_number": 12345680,
        "block_time": 15.5,
        "pending_transactions": 5000,
        "network_hashrate": "500 TH/s"
    },
    "degraded": {
        "status": "degraded",
        "block_number": 12345682,
        "block_time": 30.2,
        "pending_transactions": 12000,
        "network_hashrate": "450 TH/s"
    }
}

# Wallet response data
WALLET_RESPONSES = {
    "balance_response": {
        "eth": {"amount": "5.0", "currency": "ETH"},
        "usdc": {"amount": "10000.0", "currency": "USDC"},
        "usdt": {"amount": "5000.0", "currency": "USDT"}
    },
    "transaction_history": [
        {
            "hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "type": "swap", 
            "status": "confirmed",
            "amount": "1000.0",
            "token_in": "USDC",
            "token_out": "ETH",
            "timestamp": (datetime.now()).isoformat()
        },
        {
            "hash": "0x2345678901bcdef02345678901bcdef023456789012cdef023456789012cdef0",
            "type": "add_liquidity",
            "status": "confirmed", 
            "amount_a": "5000.0",
            "amount_b": "2.0",
            "token_a": "USDC",
            "token_b": "ETH",
            "timestamp": (datetime.now()).isoformat()
        }
    ]
}


def get_contract_abi(contract_name: str) -> List[Dict[str, Any]]:
    """Get ABI for a specific contract."""
    abis = {
        "aerodrome_pool": AERODROME_POOL_ABI,
        "aerodrome_router": AERODROME_ROUTER_ABI,
        "erc20": ERC20_ABI
    }
    return abis.get(contract_name, [])


def get_contract_address(contract_name: str) -> str:
    """Get address for a specific contract."""
    return BASE_CONTRACTS.get(contract_name, "0x0000000000000000000000000000000000000000")


def create_transaction_receipt(
    success: bool = True,
    gas_used: int = 150000,
    error: str = None
) -> Dict[str, Any]:
    """Create a transaction receipt for testing."""
    if success:
        return {
            **SUCCESSFUL_SWAP_TX,
            "gas_used": gas_used,
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            **FAILED_SWAP_TX,
            "error": error or "execution reverted",
            "timestamp": datetime.now().isoformat()
        }


def create_pool_data(
    tvl: float = 1000000.0,
    volume_24h: float = 100000.0,
    stable: bool = False
) -> Dict[str, Any]:
    """Create pool data for testing."""
    return {
        **SAMPLE_POOL_RESPONSE,
        "tvl": tvl,
        "volume_24h": volume_24h,
        "stable": stable
    }


def get_error_response(error_type: str) -> Dict[str, Any]:
    """Get error response for testing."""
    return CONTRACT_ERROR_RESPONSES.get(error_type, {
        "error": "UnknownError",
        "message": "Unknown contract error",
        "code": -32000
    })


def get_gas_price(priority: str = "standard") -> Dict[str, Any]:
    """Get gas price data for testing."""
    return GAS_PRICE_DATA.get(priority, GAS_PRICE_DATA["standard"])


def get_network_status(status: str = "healthy") -> Dict[str, Any]:
    """Get network status for testing."""
    return NETWORK_STATUS.get(status, NETWORK_STATUS["healthy"])