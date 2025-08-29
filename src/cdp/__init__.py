"""
CDP package - Complete CDP SDK integration layer

This package provides the complete CDP SDK integration for Aerodrome AI agent including:
- CDP Manager for wallet and client management
- Wallet operations for balance and token management
- Smart contract operations wrapper
- Aerodrome protocol specific operations
- Gas optimization utilities
- Comprehensive error handling

All blockchain operations must use this CDP layer exclusively.
"""

from .manager import CDPManager, MPCWallet
from .wallet import WalletOperations, TokenBalance
from .contracts import ContractOperations, ContractRegistry
from .aerodrome import AerodromeRouter, AerodromeVoter, VotingEscrow, SwapRoute, VoteAllocation
from .gas import GasOptimizer, MEVProtection, GasParams, GasEstimate
from .errors import (
    CDPError,
    WalletInitializationError,
    InsufficientBalanceError,
    TransactionError,
    ContractError,
    NetworkError,
    RateLimitError,
    GasError,
    NonceError,
    CDPErrorHandler,
    handle_cdp_error
)

__all__ = [
    # Core management
    'CDPManager',
    'MPCWallet',
    
    # Wallet operations
    'WalletOperations',
    'TokenBalance',
    
    # Contract operations
    'ContractOperations',
    'ContractRegistry',
    
    # Aerodrome operations
    'AerodromeRouter',
    'AerodromeVoter',
    'VotingEscrow',
    'SwapRoute',
    'VoteAllocation',
    
    # Gas optimization
    'GasOptimizer',
    'MEVProtection',
    'GasParams',
    'GasEstimate',
    
    # Error handling
    'CDPError',
    'WalletInitializationError',
    'InsufficientBalanceError',
    'TransactionError',
    'ContractError',
    'NetworkError',
    'RateLimitError',
    'GasError',
    'NonceError',
    'CDPErrorHandler',
    'handle_cdp_error'
]