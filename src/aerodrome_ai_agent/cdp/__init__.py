"""
CDP Integration Module - Coinbase Developer Platform SDK integration

This module provides the primary interface for all blockchain operations through
the CDP SDK. All wallet management, smart contract interactions, and transaction
operations are handled exclusively through CDP for security and reliability.

Key Components:
- CDPManager: Main CDP SDK interface and wallet management
- CDPConfig: Configuration for CDP operations
- Transaction builders for different operation types
- Event monitoring and blockchain data retrieval
"""

from .manager import CDPManager
from .config import CDPConfig
from .wallet import WalletManager
from .contracts import ContractManager

__all__ = [
    "CDPManager",
    "CDPConfig",
    "WalletManager", 
    "ContractManager",
]