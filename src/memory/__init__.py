"""
Advanced Memory System for Aerodrome AI Agent

A comprehensive memory system built on Mem0 with intelligent pruning,
pattern extraction, multi-tier storage, and high-performance caching.

Key Features:
- Multi-tier storage (hot/warm/cold/archive) with automatic migration
- Intelligent pruning based on age, relevance, and redundancy
- Advanced pattern extraction and recognition
- High-performance caching with prediction and preloading
- Comprehensive monitoring and alerting
- Specialized memory types for trading operations
"""

from .system import MemorySystem, MemoryConfig
from .operations import MemoryOperations
from .categories import MemoryCategories, SpecializedMemoryTypes
from .pruning import MemoryPruning
from .patterns import PatternExtractor
from .tiers import StorageTiers
from .cache import MemoryCache, OptimizedMemoryAccess, CacheConfig
from .metrics import MemoryMetrics

__all__ = [
    'MemorySystem',
    'MemoryConfig',
    'MemoryOperations',
    'MemoryCategories',
    'SpecializedMemoryTypes',
    'MemoryPruning',
    'PatternExtractor',
    'StorageTiers',
    'MemoryCache',
    'OptimizedMemoryAccess',
    'CacheConfig',
    'MemoryMetrics',
]

__version__ = "1.0.0"