"""
Aerodrome AI Agent - Autonomous DeFi Portfolio Manager

An AI-powered agent for managing DeFi portfolios on the Aerodrome Finance protocol
built on Base network. Utilizes LangGraph for decision-making, Mem0 for learning,
and CDP SDK for all blockchain operations.

Architecture:
- Brain: LangGraph-based state machine for cognitive operations
- Memory: Mem0-powered learning and pattern recognition
- CDP: Coinbase Developer Platform SDK for blockchain interactions
- Contracts: Aerodrome protocol contract integrations
- Utils: Shared utilities and helpers
- Config: Configuration management
"""

__version__ = "1.0.0"
__author__ = "Aerodrome AI Agent Team"
__description__ = "Autonomous AI-powered DeFi portfolio manager for Aerodrome Finance"

# Import main components for easy access
from aerodrome_ai_agent.brain import AerodromeBrain, BrainConfig, BrainState
from aerodrome_ai_agent.memory import MemorySystem, MemoryConfig
from aerodrome_ai_agent.cdp import CDPManager, CDPConfig
from aerodrome_ai_agent.config import AgentConfig

__all__ = [
    "AerodromeBrain",
    "BrainConfig", 
    "BrainState",
    "MemorySystem",
    "MemoryConfig",
    "CDPManager",
    "CDPConfig",
    "AgentConfig",
    "__version__",
    "__author__",
    "__description__",
]