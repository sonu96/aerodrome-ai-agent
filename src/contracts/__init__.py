"""
Contracts package - Aerodrome protocol contract interfaces and data

This package provides:
- Contract ABIs for all Aerodrome protocol interactions
- Contract addresses for supported networks
- Utility functions for contract management
"""

from .abis import (
    ERC20_ABI,
    ROUTER_ABI,
    POOL_ABI,
    VOTER_ABI,
    VE_AERO_ABI,
    GAUGE_ABI,
    FACTORY_ABI,
    get_abi,
    get_function_abi,
    get_event_abi
)

from .addresses import (
    AERODROME_CONTRACTS,
    TOKEN_ADDRESSES,
    POOL_ADDRESSES,
    GAUGE_ADDRESSES,
    NETWORKS,
    get_contract_address,
    get_token_address,
    get_pool_address,
    get_network_info,
    is_stable_pair,
    validate_address
)

__all__ = [
    # ABIs
    'ERC20_ABI',
    'ROUTER_ABI',
    'POOL_ABI',
    'VOTER_ABI',
    'VE_AERO_ABI',
    'GAUGE_ABI',
    'FACTORY_ABI',
    'get_abi',
    'get_function_abi',
    'get_event_abi',
    
    # Addresses and utilities
    'AERODROME_CONTRACTS',
    'TOKEN_ADDRESSES',
    'POOL_ADDRESSES',
    'GAUGE_ADDRESSES',
    'NETWORKS',
    'get_contract_address',
    'get_token_address',
    'get_pool_address',
    'get_network_info',
    'is_stable_pair',
    'validate_address'
]