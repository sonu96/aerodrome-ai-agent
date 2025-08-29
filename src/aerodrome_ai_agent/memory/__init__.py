"""
Memory System Module - Mem0-powered learning and pattern recognition

This module implements intelligent memory management with automatic pruning,
pattern extraction, and multi-tier storage for the Aerodrome AI Agent.

Key Components:
- MemorySystem: Main Mem0 integration for learning and recall
- MemoryConfig: Configuration for memory operations
- Pattern extraction and compression algorithms
- Multi-tier storage management (Hot ’ Warm ’ Cold ’ Archive)
"""

from .system import MemorySystem
from .config import MemoryConfig
from .patterns import PatternExtractor
from .pruning import MemoryPruner

__all__ = [
    "MemorySystem",
    "MemoryConfig",
    "PatternExtractor", 
    "MemoryPruner",
]