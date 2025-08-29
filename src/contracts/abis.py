"""
Contract ABIs - Store all necessary ABIs for Aerodrome protocol interactions

This module contains Application Binary Interfaces (ABIs) for all smart contracts
used in Aerodrome DeFi operations. ABIs are used by CDP SDK for type-safe
contract interactions and automatic parameter encoding/decoding.

Contains ABIs for:
- ERC20 tokens
- Aerodrome Router
- Aerodrome Pool/Pair contracts
- Aerodrome Voter contract
- veAERO (Voting Escrow)
- Aerodrome Gauge contracts
- Factory contracts
"""

from typing import List, Dict, Any

# ERC20 Token ABI
ERC20_ABI: List[Dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"}
        ],
        "name": "balanceOf",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "spender", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [
            {"internalType": "bool", "name": "", "type": "bool"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "address", "name": "spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [
            {"internalType": "bool", "name": "", "type": "bool"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [
            {"internalType": "uint8", "name": "", "type": "uint8"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [
            {"internalType": "string", "name": "", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "name",
        "outputs": [
            {"internalType": "string", "name": "", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Aerodrome Router V1 ABI
ROUTER_ABI: List[Dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
            {
                "components": [
                    {"internalType": "address", "name": "from", "type": "address"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "bool", "name": "stable", "type": "bool"}
                ],
                "internalType": "struct Route[]",
                "name": "routes",
                "type": "tuple[]"
            },
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "swapExactTokensForTokens",
        "outputs": [
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
            {
                "components": [
                    {"internalType": "address", "name": "from", "type": "address"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "bool", "name": "stable", "type": "bool"}
                ],
                "internalType": "struct Route[]",
                "name": "routes",
                "type": "tuple[]"
            },
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "swapExactETHForTokens",
        "outputs": [
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"},
            {"internalType": "bool", "name": "stable", "type": "bool"},
            {"internalType": "uint256", "name": "amountADesired", "type": "uint256"},
            {"internalType": "uint256", "name": "amountBDesired", "type": "uint256"},
            {"internalType": "uint256", "name": "amountAMin", "type": "uint256"},
            {"internalType": "uint256", "name": "amountBMin", "type": "uint256"},
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "addLiquidity",
        "outputs": [
            {"internalType": "uint256", "name": "amountA", "type": "uint256"},
            {"internalType": "uint256", "name": "amountB", "type": "uint256"},
            {"internalType": "uint256", "name": "liquidity", "type": "uint256"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"},
            {"internalType": "bool", "name": "stable", "type": "bool"},
            {"internalType": "uint256", "name": "liquidity", "type": "uint256"},
            {"internalType": "uint256", "name": "amountAMin", "type": "uint256"},
            {"internalType": "uint256", "name": "amountBMin", "type": "uint256"},
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "removeLiquidity",
        "outputs": [
            {"internalType": "uint256", "name": "amountA", "type": "uint256"},
            {"internalType": "uint256", "name": "amountB", "type": "uint256"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {
                "components": [
                    {"internalType": "address", "name": "from", "type": "address"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "bool", "name": "stable", "type": "bool"}
                ],
                "internalType": "struct Route[]",
                "name": "routes",
                "type": "tuple[]"
            }
        ],
        "name": "getAmountsOut",
        "outputs": [
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"},
            {"internalType": "bool", "name": "stable", "type": "bool"}
        ],
        "name": "pairFor",
        "outputs": [
            {"internalType": "address", "name": "pair", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Aerodrome Pool/Pair ABI
POOL_ABI: List[Dict[str, Any]] = [
    {
        "inputs": [],
        "name": "token0",
        "outputs": [
            {"internalType": "address", "name": "", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "token1",
        "outputs": [
            {"internalType": "address", "name": "", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "stable",
        "outputs": [
            {"internalType": "bool", "name": "", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"internalType": "uint256", "name": "_reserve0", "type": "uint256"},
            {"internalType": "uint256", "name": "_reserve1", "type": "uint256"},
            {"internalType": "uint256", "name": "_blockTimestampLast", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "address", "name": "tokenIn", "type": "address"}
        ],
        "name": "getAmountOut",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"}
        ],
        "name": "balanceOf",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "internalType": "address", "name": "sender", "type": "address"},
            {"indexed": false, "internalType": "uint256", "name": "amount0In", "type": "uint256"},
            {"indexed": false, "internalType": "uint256", "name": "amount1In", "type": "uint256"},
            {"indexed": false, "internalType": "uint256", "name": "amount0Out", "type": "uint256"},
            {"indexed": false, "internalType": "uint256", "name": "amount1Out", "type": "uint256"},
            {"indexed": true, "internalType": "address", "name": "to", "type": "address"}
        ],
        "name": "Swap",
        "type": "event"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "internalType": "address", "name": "sender", "type": "address"},
            {"indexed": false, "internalType": "uint256", "name": "amount0", "type": "uint256"},
            {"indexed": false, "internalType": "uint256", "name": "amount1", "type": "uint256"}
        ],
        "name": "Mint",
        "type": "event"
    }
]

# Aerodrome Voter ABI
VOTER_ABI: List[Dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"},
            {"internalType": "address[]", "name": "poolVote", "type": "address[]"},
            {"internalType": "uint256[]", "name": "weights", "type": "uint256[]"}
        ],
        "name": "vote",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
        ],
        "name": "reset",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address[]", "name": "bribes", "type": "address[]"},
            {"internalType": "address[][]", "name": "tokens", "type": "address[][]"},
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
        ],
        "name": "claimBribes",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address[]", "name": "gauges", "type": "address[]"},
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
        ],
        "name": "claimRewards",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "pool", "type": "address"}
        ],
        "name": "gauges",
        "outputs": [
            {"internalType": "address", "name": "", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"},
            {"internalType": "address", "name": "pool", "type": "address"}
        ],
        "name": "votes",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
        ],
        "name": "usedWeights",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalWeight",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# veAERO (Voting Escrow) ABI
VE_AERO_ABI: List[Dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "value", "type": "uint256"},
            {"internalType": "uint256", "name": "lockDuration", "type": "uint256"}
        ],
        "name": "create_lock",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"},
            {"internalType": "uint256", "name": "value", "type": "uint256"}
        ],
        "name": "increase_amount",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"},
            {"internalType": "uint256", "name": "lockDuration", "type": "uint256"}
        ],
        "name": "increase_unlock_time",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
        ],
        "name": "withdraw",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
        ],
        "name": "balanceOfNFT",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
        ],
        "name": "locked",
        "outputs": [
            {
                "components": [
                    {"internalType": "int128", "name": "amount", "type": "int128"},
                    {"internalType": "uint256", "name": "end", "type": "uint256"}
                ],
                "internalType": "struct LockedBalance",
                "name": "",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"}
        ],
        "name": "balanceOf",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "uint256", "name": "index", "type": "uint256"}
        ],
        "name": "tokenOfOwnerByIndex",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Aerodrome Gauge ABI
GAUGE_ABI: List[Dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
        ],
        "name": "deposit",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amount", "type": "uint256"}
        ],
        "name": "withdraw",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"}
        ],
        "name": "getReward",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"}
        ],
        "name": "balanceOf",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"}
        ],
        "name": "earned",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "stakingToken",
        "outputs": [
            {"internalType": "address", "name": "", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "rewardsToken",
        "outputs": [
            {"internalType": "address", "name": "", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Aerodrome Factory ABI
FACTORY_ABI: List[Dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"},
            {"internalType": "bool", "name": "stable", "type": "bool"}
        ],
        "name": "createPair",
        "outputs": [
            {"internalType": "address", "name": "pair", "type": "address"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"},
            {"internalType": "bool", "name": "stable", "type": "bool"}
        ],
        "name": "getPair",
        "outputs": [
            {"internalType": "address", "name": "pair", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "allPairsLength",
        "outputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "name": "allPairs",
        "outputs": [
            {"internalType": "address", "name": "", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Multicall ABI for batch operations
MULTICALL_ABI: List[Dict[str, Any]] = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "address", "name": "target", "type": "address"},
                    {"internalType": "bytes", "name": "callData", "type": "bytes"}
                ],
                "internalType": "struct Multicall.Call[]",
                "name": "calls",
                "type": "tuple[]"
            }
        ],
        "name": "aggregate",
        "outputs": [
            {"internalType": "uint256", "name": "blockNumber", "type": "uint256"},
            {"internalType": "bytes[]", "name": "returnData", "type": "bytes[]"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# WETH ABI for wrapped ETH operations
WETH_ABI: List[Dict[str, Any]] = [
    {
        "inputs": [],
        "name": "deposit",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "wad", "type": "uint256"}
        ],
        "name": "withdraw",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    # Include standard ERC20 functions
    *ERC20_ABI
]

# Convenience dictionary mapping contract types to their ABIs
ABI_MAP: Dict[str, List[Dict[str, Any]]] = {
    'ERC20': ERC20_ABI,
    'ROUTER': ROUTER_ABI,
    'POOL': POOL_ABI,
    'PAIR': POOL_ABI,  # Pool and Pair are the same in Aerodrome
    'VOTER': VOTER_ABI,
    'VE_AERO': VE_AERO_ABI,
    'GAUGE': GAUGE_ABI,
    'FACTORY': FACTORY_ABI,
    'MULTICALL': MULTICALL_ABI,
    'WETH': WETH_ABI
}

def get_abi(contract_type: str) -> List[Dict[str, Any]]:
    """
    Get ABI for a specific contract type.
    
    Args:
        contract_type: Type of contract ('ERC20', 'ROUTER', 'POOL', etc.)
        
    Returns:
        List of ABI function/event definitions
        
    Raises:
        KeyError: If contract_type is not found
    """
    contract_type = contract_type.upper()
    if contract_type not in ABI_MAP:
        raise KeyError(f"ABI not found for contract type: {contract_type}")
    
    return ABI_MAP[contract_type]

def get_function_abi(contract_type: str, function_name: str) -> Dict[str, Any]:
    """
    Get specific function ABI from contract.
    
    Args:
        contract_type: Type of contract
        function_name: Name of the function
        
    Returns:
        Function ABI definition
        
    Raises:
        KeyError: If contract type or function not found
    """
    abi = get_abi(contract_type)
    
    for item in abi:
        if item.get('type') == 'function' and item.get('name') == function_name:
            return item
    
    raise KeyError(f"Function '{function_name}' not found in {contract_type} ABI")

def get_event_abi(contract_type: str, event_name: str) -> Dict[str, Any]:
    """
    Get specific event ABI from contract.
    
    Args:
        contract_type: Type of contract
        event_name: Name of the event
        
    Returns:
        Event ABI definition
        
    Raises:
        KeyError: If contract type or event not found
    """
    abi = get_abi(contract_type)
    
    for item in abi:
        if item.get('type') == 'event' and item.get('name') == event_name:
            return item
    
    raise KeyError(f"Event '{event_name}' not found in {contract_type} ABI")

def validate_abi_function_inputs(
    contract_type: str, 
    function_name: str, 
    inputs: Dict[str, Any]
) -> bool:
    """
    Validate function inputs against ABI specification.
    
    Args:
        contract_type: Type of contract
        function_name: Name of the function
        inputs: Input parameters to validate
        
    Returns:
        True if inputs are valid
        
    Raises:
        ValueError: If inputs don't match ABI specification
    """
    try:
        function_abi = get_function_abi(contract_type, function_name)
        expected_inputs = {inp['name']: inp['type'] for inp in function_abi.get('inputs', [])}
        
        # Check all required inputs are provided
        for param_name, param_type in expected_inputs.items():
            if param_name not in inputs:
                raise ValueError(f"Missing required parameter '{param_name}' of type '{param_type}'")
        
        # Check no extra inputs are provided
        for param_name in inputs:
            if param_name not in expected_inputs:
                raise ValueError(f"Unexpected parameter '{param_name}'")
        
        return True
        
    except KeyError as e:
        raise ValueError(f"ABI validation failed: {str(e)}")

# Export commonly used ABIs at module level
__all__ = [
    'ERC20_ABI',
    'ROUTER_ABI', 
    'POOL_ABI',
    'VOTER_ABI',
    'VE_AERO_ABI',
    'GAUGE_ABI',
    'FACTORY_ABI',
    'MULTICALL_ABI',
    'WETH_ABI',
    'ABI_MAP',
    'get_abi',
    'get_function_abi',
    'get_event_abi',
    'validate_abi_function_inputs'
]