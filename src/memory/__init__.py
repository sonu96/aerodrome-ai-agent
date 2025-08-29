"""
Memory management system for the Aerodrome AI Agent.

This package provides sophisticated memory pruning, categorization, and management
capabilities using Mem0's latest features including graph memory, batch operations,
and advanced filtering.
"""

from .memory_categories import (
    MemoryCategory,
    RetentionPolicy,
    MemoryCategoryConfig,
    MemoryMetadata
)
from .mem0_client import EnhancedMem0Client
from .pruning_engine import (
    MemoryPruningEngine,
    PruningStats,
    ConsolidationCandidate
)

__all__ = [
    "MemoryCategory",
    "RetentionPolicy", 
    "MemoryCategoryConfig",
    "MemoryMetadata",
    "EnhancedMem0Client",
    "MemoryPruningEngine",
    "PruningStats",
    "ConsolidationCandidate"
]

__version__ = "1.0.0"