"""
Memory data fixtures for testing the memory system.

Contains sample memory entries, patterns, and learning scenarios
for testing memory operations.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List


# Sample experience memories
SUCCESSFUL_SWAP_MEMORY = {
    "id": "memory_swap_001",
    "type": "experience",
    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
    "experience": {
        "action_type": "SWAP",
        "pool": "0x1234567890abcdef1234567890abcdef12345678",
        "token_in": "USDC",
        "token_out": "WETH",
        "amount": 1000.0,
        "confidence": 0.85,
        "gas_price": 20.0,
        "slippage": 0.005,
        "market_conditions": {
            "volatility": 0.08,
            "volume": 2000000.0,
            "trending": "bullish"
        }
    },
    "outcome": {
        "success": True,
        "profit": 25.50,
        "actual_output": 0.398,
        "gas_used": 125000,
        "execution_time": 15.5,
        "confidence": 0.92
    }
}

FAILED_SWAP_MEMORY = {
    "id": "memory_swap_002",
    "type": "experience",
    "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
    "experience": {
        "action_type": "SWAP",
        "pool": "0x2345678901bcdef02345678901bcdef023456789",
        "token_in": "WETH",
        "token_out": "USDC",
        "amount": 0.5,
        "confidence": 0.65,
        "gas_price": 45.0,
        "slippage": 0.02,
        "market_conditions": {
            "volatility": 0.35,
            "volume": 500000.0,
            "trending": "volatile"
        }
    },
    "outcome": {
        "success": False,
        "profit": -15.25,
        "actual_output": 0.0,
        "gas_used": 0,
        "execution_time": 0.0,
        "error": "SlippageExceeded",
        "confidence": 0.45
    }
}

LIQUIDITY_ADDITION_MEMORY = {
    "id": "memory_lp_001",
    "type": "experience", 
    "timestamp": (datetime.now() - timedelta(hours=8)).isoformat(),
    "experience": {
        "action_type": "ADD_LIQUIDITY",
        "pool": "0x3456789012cdef13456789012cdef1345678901",
        "token_a": "USDC",
        "token_b": "USDT",
        "amount_a": 5000.0,
        "amount_b": 5000.0,
        "confidence": 0.78,
        "gas_price": 18.0,
        "market_conditions": {
            "volatility": 0.02,
            "volume": 10000000.0,
            "trending": "stable"
        }
    },
    "outcome": {
        "success": True,
        "profit": 0.0,  # No immediate profit for LP
        "lp_tokens": 9995.0,
        "gas_used": 180000,
        "execution_time": 25.8,
        "apr": 0.085,
        "confidence": 0.82
    }
}

# Pattern examples
SUCCESSFUL_SWAP_PATTERN = {
    "id": "pattern_001",
    "type": "behavioral_pattern",
    "pattern_type": "successful_swap_conditions",
    "occurrences": 8,
    "confidence": 0.87,
    "conditions": {
        "market_volatility": {"min": 0.05, "max": 0.15},
        "gas_price": {"max": 30.0},
        "volume_threshold": {"min": 1000000.0},
        "confidence_threshold": {"min": 0.75},
        "slippage": {"max": 0.01}
    },
    "outcomes": {
        "success_rate": 0.875,
        "avg_profit": 22.3,
        "avg_execution_time": 18.2,
        "profit_range": {"min": 8.5, "max": 45.2}
    },
    "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
    "last_updated": (datetime.now() - timedelta(hours=3)).isoformat()
}

HIGH_GAS_FAILURE_PATTERN = {
    "id": "pattern_002",
    "type": "failure_pattern",
    "pattern_type": "high_gas_failures",
    "occurrences": 5,
    "confidence": 0.93,
    "conditions": {
        "gas_price": {"min": 40.0},
        "market_volatility": {"min": 0.25},
        "action_type": "SWAP"
    },
    "outcomes": {
        "success_rate": 0.2,
        "avg_loss": -18.7,
        "common_errors": ["HighGasPriceError", "SlippageExceeded"],
        "recommendation": "Wait for gas prices to normalize"
    },
    "created_at": (datetime.now() - timedelta(days=2)).isoformat(),
    "last_updated": (datetime.now() - timedelta(hours=1)).isoformat()
}

STABLE_PAIR_LP_PATTERN = {
    "id": "pattern_003",
    "type": "strategy_pattern",
    "pattern_type": "stable_pair_lp_strategy",
    "occurrences": 12,
    "confidence": 0.91,
    "conditions": {
        "token_pair_type": "stable_stable",
        "tvl": {"min": 5000000.0},
        "apr": {"min": 0.06},
        "pool_age_days": {"min": 30}
    },
    "outcomes": {
        "success_rate": 0.92,
        "avg_apr": 0.084,
        "impermanent_loss": 0.003,
        "hold_duration_days": 45.2
    },
    "created_at": (datetime.now() - timedelta(days=5)).isoformat(),
    "last_updated": (datetime.now() - timedelta(hours=6)).isoformat()
}

# Memory categories for testing
MEMORY_CATEGORIES = {
    "experience": [
        SUCCESSFUL_SWAP_MEMORY,
        FAILED_SWAP_MEMORY,
        LIQUIDITY_ADDITION_MEMORY
    ],
    "patterns": [
        SUCCESSFUL_SWAP_PATTERN,
        HIGH_GAS_FAILURE_PATTERN,
        STABLE_PAIR_LP_PATTERN
    ]
}

# Realistic memory collections for different time periods
def generate_memory_sequence(days: int = 7) -> List[Dict[str, Any]]:
    """Generate a sequence of memories over a time period."""
    memories = []
    base_time = datetime.now() - timedelta(days=days)
    
    # Generate diverse memory types
    memory_templates = [
        {
            "type": "SWAP",
            "success_rate": 0.75,
            "avg_profit": 18.5,
            "avg_confidence": 0.72
        },
        {
            "type": "ADD_LIQUIDITY",
            "success_rate": 0.85,
            "avg_profit": 0.0,
            "avg_confidence": 0.68
        },
        {
            "type": "REMOVE_LIQUIDITY",
            "success_rate": 0.90,
            "avg_profit": -5.2,
            "avg_confidence": 0.75
        }
    ]
    
    import random
    
    for day in range(days):
        # 2-5 memories per day
        daily_memories = random.randint(2, 5)
        
        for memory_idx in range(daily_memories):
            template = random.choice(memory_templates)
            memory_time = base_time + timedelta(
                days=day,
                hours=random.randint(8, 20),
                minutes=random.randint(0, 59)
            )
            
            success = random.random() < template["success_rate"]
            
            memory = {
                "id": f"memory_{day}_{memory_idx}",
                "type": "experience",
                "timestamp": memory_time.isoformat(),
                "experience": {
                    "action_type": template["type"],
                    "pool": f"0x{''.join([random.choice('0123456789abcdef') for _ in range(40)])}",
                    "amount": random.uniform(100, 5000),
                    "confidence": max(0.5, min(1.0, random.gauss(template["avg_confidence"], 0.1))),
                    "gas_price": random.uniform(15, 50),
                    "market_conditions": {
                        "volatility": random.uniform(0.02, 0.4),
                        "volume": random.uniform(500000, 5000000),
                        "trending": random.choice(["bullish", "bearish", "stable", "volatile"])
                    }
                },
                "outcome": {
                    "success": success,
                    "profit": (random.gauss(template["avg_profit"], 10) 
                              if success else -random.uniform(5, 25)),
                    "confidence": random.uniform(0.5, 0.95) if success else random.uniform(0.2, 0.6),
                    "gas_used": random.randint(100000, 250000) if success else 0,
                    "execution_time": random.uniform(10, 30) if success else 0
                }
            }
            
            if not success:
                memory["outcome"]["error"] = random.choice([
                    "SlippageExceeded", "InsufficientBalance", 
                    "HighGasPriceError", "NetworkError"
                ])
            
            memories.append(memory)
    
    return memories


# Test scenarios for memory operations
MEMORY_QUERY_SCENARIOS = [
    {
        "name": "recent_swaps",
        "context": {
            "action_type": "SWAP",
            "timeframe": "recent",
            "success_only": False
        },
        "expected_count": 5,
        "min_relevance": 0.8
    },
    {
        "name": "high_confidence_trades",
        "context": {
            "confidence_threshold": 0.8,
            "success_only": True
        },
        "expected_count": 3,
        "min_relevance": 0.75
    },
    {
        "name": "volatile_market_experiences",
        "context": {
            "market_conditions": {"volatile": True},
            "timeframe": "week"
        },
        "expected_count": 4,
        "min_relevance": 0.7
    }
]

# Memory system configuration for testing
TEST_MEMORY_CONFIG = {
    "user_id": "test_agent",
    "max_memories": 1000,
    "pattern_threshold": 3,
    "relevance_threshold": 0.6,
    "pruning_enabled": True,
    "hot_tier_days": 1,
    "warm_tier_days": 7,
    "cold_tier_days": 30,
    "max_memory_age_days": 90
}

# Error scenarios for memory operations
MEMORY_ERROR_SCENARIOS = [
    {
        "error_type": "ConnectionError",
        "operation": "store_memory",
        "message": "Failed to connect to vector database",
        "retryable": True
    },
    {
        "error_type": "ValidationError", 
        "operation": "recall_memories",
        "message": "Invalid memory query format",
        "retryable": False
    },
    {
        "error_type": "QuotaExceeded",
        "operation": "extract_patterns",
        "message": "Memory storage quota exceeded",
        "retryable": False
    }
]


def get_memories_by_type(memory_type: str) -> List[Dict[str, Any]]:
    """Get memories filtered by type."""
    return [mem for mem in MEMORY_CATEGORIES.get("experience", []) 
            if mem["experience"]["action_type"] == memory_type]


def get_memories_by_success(successful: bool) -> List[Dict[str, Any]]:
    """Get memories filtered by success status."""
    return [mem for mem in MEMORY_CATEGORIES.get("experience", [])
            if mem["outcome"]["success"] == successful]


def get_patterns_by_type(pattern_type: str) -> List[Dict[str, Any]]:
    """Get patterns filtered by type.""" 
    return [pattern for pattern in MEMORY_CATEGORIES.get("patterns", [])
            if pattern["pattern_type"] == pattern_type]