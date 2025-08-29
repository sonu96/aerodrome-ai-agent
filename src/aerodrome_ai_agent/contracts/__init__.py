"""
Contracts Module - Aerodrome protocol contract interfaces

This module contains ABI definitions and utilities for interacting with
Aerodrome Finance protocol contracts on Base network.

Key Components:
- Contract ABIs for all Aerodrome protocol contracts
- Utility functions for contract interaction
- Contract address constants
- Data processing helpers
"""

from .abis import (
    ROUTER_ABI,
    POOL_ABI, 
    FACTORY_ABI,
    VOTER_ABI,
    GAUGE_ABI,
    BRIBE_ABI
)

from .addresses import (
    AERODROME_ROUTER,
    AERODROME_FACTORY,
    AERODROME_VOTER,
    BASE_TOKENS
)

from .utils import (
    encode_function_call,
    decode_function_result,
    get_pool_address,
    calculate_amounts_out
)

__all__ = [
    # ABIs
    "ROUTER_ABI",
    "POOL_ABI",
    "FACTORY_ABI", 
    "VOTER_ABI",
    "GAUGE_ABI",
    "BRIBE_ABI",
    
    # Addresses
    "AERODROME_ROUTER",
    "AERODROME_FACTORY", 
    "AERODROME_VOTER",
    "BASE_TOKENS",
    
    # Utils
    "encode_function_call",
    "decode_function_result",
    "get_pool_address",
    "calculate_amounts_out",
]