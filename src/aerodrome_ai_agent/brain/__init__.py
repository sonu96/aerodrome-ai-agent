"""
Brain Module - LangGraph-based cognitive system for the Aerodrome AI Agent

This module implements the core decision-making brain using LangGraph state machine
architecture. The brain handles market observation, memory recall, opportunity analysis,
risk assessment, decision making, and execution planning.

Key Components:
- AerodromeBrain: Main brain class with LangGraph state machine
- BrainConfig: Configuration settings for brain operations  
- BrainState: TypedDict defining complete state representation
- Node implementations for each cognitive function
"""

from .core import AerodromeBrain
from .config import BrainConfig
from .state import BrainState
from .nodes import *

__all__ = [
    "AerodromeBrain",
    "BrainConfig", 
    "BrainState",
]