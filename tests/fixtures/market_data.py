"""
Market data fixtures for testing.

Contains realistic market data scenarios for testing brain decision-making
and opportunity detection.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List


# Base market conditions
STABLE_MARKET = {
    "eth_price": 2500.0,
    "base_eth_price": 2498.5,
    "volatility": 0.05,
    "gas_price": 15.0,
    "volume_24h": 1000000.0,
    "trending": "neutral",
    "fear_greed_index": 50
}

VOLATILE_MARKET = {
    "eth_price": 2400.0,
    "base_eth_price": 2410.0,
    "volatility": 0.35,
    "gas_price": 45.0,
    "volume_24h": 2500000.0,
    "trending": "bearish",
    "fear_greed_index": 25
}

BULL_MARKET = {
    "eth_price": 2800.0,
    "base_eth_price": 2795.0,
    "volatility": 0.15,
    "gas_price": 25.0,
    "volume_24h": 3000000.0,
    "trending": "bullish",
    "fear_greed_index": 75
}

# Sample pool data for different scenarios
SAMPLE_POOLS = [
    {
        "address": "0x1234567890abcdef1234567890abcdef12345678",
        "token0": {
            "address": "0xA0b86a33E6441c8C4C0a27b2B7fE9A7b6e3a1234",
            "symbol": "USDC",
            "decimals": 6,
            "balance": "1000000000000"
        },
        "token1": {
            "address": "0x4200000000000000000000000000000000000006",
            "symbol": "WETH",
            "decimals": 18,
            "balance": "400000000000000000000"
        },
        "fee": 3000,
        "tvl": 2500000.0,
        "volume_24h": 500000.0,
        "fee_growth": 0.015,
        "price": 2500.0,
        "liquidity": "75000000000000000000000",
        "apr": 0.125,
        "stable": False
    },
    {
        "address": "0x2345678901bcdef02345678901bcdef023456789",
        "token0": {
            "address": "0xA0b86a33E6441c8C4C0a27b2B7fE9A7b6e3a1234",
            "symbol": "USDC",
            "decimals": 6,
            "balance": "5000000000000"
        },
        "token1": {
            "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "symbol": "USDT",
            "decimals": 6,
            "balance": "5000000000000"
        },
        "fee": 500,
        "tvl": 10000000.0,
        "volume_24h": 2000000.0,
        "fee_growth": 0.08,
        "price": 1.0001,
        "liquidity": "100000000000000000000000",
        "apr": 0.08,
        "stable": True
    }
]

# Opportunity scenarios
HIGH_CONFIDENCE_OPPORTUNITY = {
    "type": "SWAP",
    "pool_address": "0x1234567890abcdef1234567890abcdef12345678",
    "token_in": "USDC",
    "token_out": "WETH",
    "amount_in": 1000.0,
    "expected_output": 0.398,
    "score": 0.92,
    "confidence": 0.89,
    "gas_estimate": 125000,
    "slippage": 0.005,
    "price_impact": 0.002,
    "expected_profit": 25.50,
    "risk_score": 0.15,
    "rationale": "Arbitrage opportunity detected with low slippage and high confidence"
}

MEDIUM_CONFIDENCE_OPPORTUNITY = {
    "type": "ADD_LIQUIDITY",
    "pool_address": "0x2345678901bcdef02345678901bcdef023456789",
    "token_a": "USDC",
    "token_b": "USDT",
    "amount_a": 5000.0,
    "amount_b": 5000.0,
    "expected_lp_tokens": 10000.0,
    "score": 0.72,
    "confidence": 0.68,
    "gas_estimate": 180000,
    "expected_apr": 0.08,
    "impermanent_loss_risk": 0.02,
    "expected_profit": 40.0,
    "risk_score": 0.25,
    "rationale": "Stable pair with decent APR and low IL risk"
}

LOW_CONFIDENCE_OPPORTUNITY = {
    "type": "REMOVE_LIQUIDITY",
    "pool_address": "0x3456789012cdef13456789012cdef1345678901",
    "lp_tokens": 1000.0,
    "expected_token_a": 500.0,
    "expected_token_b": 0.2,
    "score": 0.45,
    "confidence": 0.52,
    "gas_estimate": 200000,
    "slippage": 0.015,
    "expected_profit": -5.0,
    "risk_score": 0.65,
    "rationale": "Position showing losses, consider exit to prevent further decline"
}

# Historical data patterns
HISTORICAL_PRICES = [
    {"timestamp": datetime.now() - timedelta(hours=24), "eth_price": 2450.0},
    {"timestamp": datetime.now() - timedelta(hours=23), "eth_price": 2460.0},
    {"timestamp": datetime.now() - timedelta(hours=22), "eth_price": 2455.0},
    {"timestamp": datetime.now() - timedelta(hours=21), "eth_price": 2470.0},
    {"timestamp": datetime.now() - timedelta(hours=20), "eth_price": 2485.0},
    {"timestamp": datetime.now() - timedelta(hours=19), "eth_price": 2490.0},
    {"timestamp": datetime.now() - timedelta(hours=18), "eth_price": 2500.0},
    {"timestamp": datetime.now() - timedelta(hours=17), "eth_price": 2495.0},
    {"timestamp": datetime.now() - timedelta(hours=16), "eth_price": 2505.0},
    {"timestamp": datetime.now() - timedelta(hours=15), "eth_price": 2510.0},
    {"timestamp": datetime.now() - timedelta(hours=14), "eth_price": 2515.0},
    {"timestamp": datetime.now() - timedelta(hours=13), "eth_price": 2508.0},
    {"timestamp": datetime.now() - timedelta(hours=12), "eth_price": 2502.0},
    {"timestamp": datetime.now() - timedelta(hours=11), "eth_price": 2498.0},
    {"timestamp": datetime.now() - timedelta(hours=10), "eth_price": 2505.0},
    {"timestamp": datetime.now() - timedelta(hours=9), "eth_price": 2512.0},
    {"timestamp": datetime.now() - timedelta(hours=8), "eth_price": 2520.0},
    {"timestamp": datetime.now() - timedelta(hours=7), "eth_price": 2518.0},
    {"timestamp": datetime.now() - timedelta(hours=6), "eth_price": 2525.0},
    {"timestamp": datetime.now() - timedelta(hours=5), "eth_price": 2530.0},
    {"timestamp": datetime.now() - timedelta(hours=4), "eth_price": 2528.0},
    {"timestamp": datetime.now() - timedelta(hours=3), "eth_price": 2535.0},
    {"timestamp": datetime.now() - timedelta(hours=2), "eth_price": 2540.0},
    {"timestamp": datetime.now() - timedelta(hours=1), "eth_price": 2545.0},
    {"timestamp": datetime.now(), "eth_price": 2550.0}
]

# Wallet scenarios
WHALE_WALLET = {
    "eth": 100.0,
    "usdc": 250000.0,
    "usdt": 50000.0,
    "wbtc": 5.0,
    "aerodrome_tokens": 50000.0,
    "lp_tokens": {
        "eth_usdc": 10000.0,
        "usdc_usdt": 25000.0
    }
}

MODERATE_WALLET = {
    "eth": 10.0,
    "usdc": 25000.0,
    "usdt": 5000.0,
    "wbtc": 0.5,
    "aerodrome_tokens": 5000.0,
    "lp_tokens": {
        "eth_usdc": 1000.0,
        "usdc_usdt": 2500.0
    }
}

SMALL_WALLET = {
    "eth": 1.0,
    "usdc": 2500.0,
    "usdt": 500.0,
    "wbtc": 0.05,
    "aerodrome_tokens": 500.0,
    "lp_tokens": {}
}

# Error scenarios
NETWORK_ERROR_RESPONSE = {
    "error": "NetworkError",
    "message": "Connection timeout to blockchain node",
    "code": 503,
    "retryable": True
}

INSUFFICIENT_BALANCE_ERROR = {
    "error": "InsufficientBalance",
    "message": "Insufficient token balance for transaction",
    "code": 400,
    "retryable": False
}

HIGH_GAS_PRICE_ERROR = {
    "error": "HighGasPriceError", 
    "message": "Gas price exceeds maximum allowed threshold",
    "code": 400,
    "retryable": True
}

SLIPPAGE_ERROR = {
    "error": "SlippageExceeded",
    "message": "Transaction would exceed maximum slippage tolerance",
    "code": 400,
    "retryable": False
}


def get_market_scenario(scenario_name: str) -> Dict[str, Any]:
    """Get market data for a specific scenario."""
    scenarios = {
        "stable": STABLE_MARKET,
        "volatile": VOLATILE_MARKET,
        "bull": BULL_MARKET
    }
    return scenarios.get(scenario_name, STABLE_MARKET)


def get_opportunity_by_confidence(confidence_level: str) -> Dict[str, Any]:
    """Get opportunity data by confidence level."""
    opportunities = {
        "high": HIGH_CONFIDENCE_OPPORTUNITY,
        "medium": MEDIUM_CONFIDENCE_OPPORTUNITY,
        "low": LOW_CONFIDENCE_OPPORTUNITY
    }
    return opportunities.get(confidence_level, MEDIUM_CONFIDENCE_OPPORTUNITY)


def get_wallet_scenario(size: str) -> Dict[str, Any]:
    """Get wallet data for different portfolio sizes."""
    wallets = {
        "whale": WHALE_WALLET,
        "moderate": MODERATE_WALLET,
        "small": SMALL_WALLET
    }
    return wallets.get(size, MODERATE_WALLET)


def generate_price_series(
    base_price: float,
    length: int,
    volatility: float = 0.02
) -> List[Dict[str, Any]]:
    """Generate a price series for testing."""
    import random
    
    prices = []
    current_price = base_price
    
    for i in range(length):
        # Add some randomness
        change = random.gauss(0, volatility)
        current_price *= (1 + change)
        
        prices.append({
            "timestamp": datetime.now() - timedelta(hours=length-i),
            "price": round(current_price, 2)
        })
    
    return prices