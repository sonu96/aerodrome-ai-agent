"""
Memory category definitions for the Aerodrome AI Agent memory system.

This module defines memory categories, retention policies, decay rates, and thresholds
for different types of information stored in the Mem0 system.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class MemoryCategory(Enum):
    """Memory category enumeration."""
    PROTOCOL_CONSTANTS = "protocol_constants"
    POOL_PERFORMANCE = "pool_performance" 
    VOTING_PATTERNS = "voting_patterns"
    MARKET_CORRELATIONS = "market_correlations"
    SPECULATIVE_INSIGHTS = "speculative_insights"
    USER_PREFERENCES = "user_preferences"
    TRADING_STRATEGIES = "trading_strategies"
    RISK_ASSESSMENTS = "risk_assessments"
    HISTORICAL_EVENTS = "historical_events"
    TECHNICAL_ANALYSIS = "technical_analysis"


@dataclass
class RetentionPolicy:
    """Retention policy configuration for a memory category."""
    max_age: timedelta
    min_confidence_threshold: float
    decay_rate: float  # How quickly confidence decays over time
    consolidation_threshold: int  # Min number of similar memories to trigger consolidation
    batch_size: int  # Optimal batch size for operations
    priority_weight: float  # Higher weight = higher priority for retention
    requires_graph_pruning: bool = False
    custom_filters: Optional[Dict[str, Any]] = None


class MemoryCategoryConfig:
    """Configuration manager for memory categories and their retention policies."""
    
    # Category-specific retention policies
    CATEGORY_POLICIES: Dict[MemoryCategory, RetentionPolicy] = {
        MemoryCategory.PROTOCOL_CONSTANTS: RetentionPolicy(
            max_age=timedelta(days=365),  # Very stable information
            min_confidence_threshold=0.9,
            decay_rate=0.001,  # Extremely slow decay
            consolidation_threshold=3,
            batch_size=100,
            priority_weight=1.0,
            requires_graph_pruning=False,
            custom_filters={"stability_score": {"$gte": 0.8}}
        ),
        
        MemoryCategory.POOL_PERFORMANCE: RetentionPolicy(
            max_age=timedelta(days=90),  # Performance data becomes stale
            min_confidence_threshold=0.7,
            decay_rate=0.01,  # Moderate decay
            consolidation_threshold=5,
            batch_size=200,
            priority_weight=0.8,
            requires_graph_pruning=True,
            custom_filters={"pool_tvl": {"$gte": 1000}}
        ),
        
        MemoryCategory.VOTING_PATTERNS: RetentionPolicy(
            max_age=timedelta(days=180),  # Voting patterns have medium-term relevance
            min_confidence_threshold=0.75,
            decay_rate=0.005,  # Slow decay
            consolidation_threshold=4,
            batch_size=150,
            priority_weight=0.85,
            requires_graph_pruning=True,
            custom_filters={"vote_weight": {"$gte": 100}}
        ),
        
        MemoryCategory.MARKET_CORRELATIONS: RetentionPolicy(
            max_age=timedelta(days=60),  # Market correlations change frequently
            min_confidence_threshold=0.8,
            decay_rate=0.02,  # Fast decay due to market volatility
            consolidation_threshold=6,
            batch_size=250,
            priority_weight=0.9,
            requires_graph_pruning=True,
            custom_filters={"correlation_strength": {"$gte": 0.6}}
        ),
        
        MemoryCategory.SPECULATIVE_INSIGHTS: RetentionPolicy(
            max_age=timedelta(days=30),  # Speculative insights are short-lived
            min_confidence_threshold=0.6,
            decay_rate=0.05,  # Very fast decay
            consolidation_threshold=8,
            batch_size=100,
            priority_weight=0.5,
            requires_graph_pruning=False,
            custom_filters={"speculation_level": {"$lte": 0.8}}
        ),
        
        MemoryCategory.USER_PREFERENCES: RetentionPolicy(
            max_age=timedelta(days=365),  # User preferences are long-term
            min_confidence_threshold=0.8,
            decay_rate=0.002,  # Very slow decay
            consolidation_threshold=2,
            batch_size=50,
            priority_weight=0.95,
            requires_graph_pruning=False,
            custom_filters={"user_activity": {"$gte": 0.5}}
        ),
        
        MemoryCategory.TRADING_STRATEGIES: RetentionPolicy(
            max_age=timedelta(days=120),  # Strategies need periodic review
            min_confidence_threshold=0.85,
            decay_rate=0.008,  # Moderate decay
            consolidation_threshold=3,
            batch_size=75,
            priority_weight=0.92,
            requires_graph_pruning=True,
            custom_filters={"success_rate": {"$gte": 0.6}}
        ),
        
        MemoryCategory.RISK_ASSESSMENTS: RetentionPolicy(
            max_age=timedelta(days=45),  # Risk profiles change with market conditions
            min_confidence_threshold=0.88,
            decay_rate=0.015,  # Fast decay due to changing risk landscape
            consolidation_threshold=4,
            batch_size=100,
            priority_weight=0.88,
            requires_graph_pruning=True,
            custom_filters={"risk_score": {"$lte": 0.8}}
        ),
        
        MemoryCategory.HISTORICAL_EVENTS: RetentionPolicy(
            max_age=timedelta(days=720),  # Historical events are valuable long-term
            min_confidence_threshold=0.9,
            decay_rate=0.0005,  # Extremely slow decay
            consolidation_threshold=2,
            batch_size=150,
            priority_weight=0.75,
            requires_graph_pruning=True,
            custom_filters={"event_impact": {"$gte": 0.7}}
        ),
        
        MemoryCategory.TECHNICAL_ANALYSIS: RetentionPolicy(
            max_age=timedelta(days=30),  # Technical analysis is short-term
            min_confidence_threshold=0.75,
            decay_rate=0.03,  # Fast decay due to changing technical conditions
            consolidation_threshold=5,
            batch_size=200,
            priority_weight=0.7,
            requires_graph_pruning=True,
            custom_filters={"timeframe": {"$in": ["1h", "4h", "1d"]}}
        )
    }
    
    # Tiered pruning thresholds
    PRUNING_TIERS = {
        "hourly": {
            "confidence_threshold": 0.3,
            "max_memories": 10000,
            "categories": [
                MemoryCategory.SPECULATIVE_INSIGHTS,
                MemoryCategory.TECHNICAL_ANALYSIS
            ]
        },
        "daily": {
            "confidence_threshold": 0.5,
            "max_memories": 50000,
            "categories": [
                MemoryCategory.MARKET_CORRELATIONS,
                MemoryCategory.POOL_PERFORMANCE,
                MemoryCategory.RISK_ASSESSMENTS
            ]
        },
        "weekly": {
            "confidence_threshold": 0.7,
            "max_memories": 100000,
            "categories": [
                MemoryCategory.VOTING_PATTERNS,
                MemoryCategory.TRADING_STRATEGIES
            ]
        },
        "monthly": {
            "confidence_threshold": 0.8,
            "max_memories": 200000,
            "categories": [
                MemoryCategory.USER_PREFERENCES,
                MemoryCategory.PROTOCOL_CONSTANTS,
                MemoryCategory.HISTORICAL_EVENTS
            ]
        }
    }
    
    @classmethod
    def get_policy(cls, category: MemoryCategory) -> RetentionPolicy:
        """Get retention policy for a specific category."""
        policy = cls.CATEGORY_POLICIES.get(category)
        if not policy:
            logger.warning(f"No policy found for category {category}, using default")
            return cls._get_default_policy()
        return policy
    
    @classmethod
    def _get_default_policy(cls) -> RetentionPolicy:
        """Get default retention policy."""
        return RetentionPolicy(
            max_age=timedelta(days=30),
            min_confidence_threshold=0.7,
            decay_rate=0.01,
            consolidation_threshold=5,
            batch_size=100,
            priority_weight=0.5,
            requires_graph_pruning=False
        )
    
    @classmethod
    def get_categories_by_tier(cls, tier: str) -> list[MemoryCategory]:
        """Get memory categories for a specific pruning tier."""
        tier_config = cls.PRUNING_TIERS.get(tier, {})
        return tier_config.get("categories", [])
    
    @classmethod
    def get_tier_threshold(cls, tier: str) -> float:
        """Get confidence threshold for a specific pruning tier."""
        tier_config = cls.PRUNING_TIERS.get(tier, {})
        return tier_config.get("confidence_threshold", 0.5)
    
    @classmethod
    def get_tier_max_memories(cls, tier: str) -> int:
        """Get maximum memories limit for a specific pruning tier."""
        tier_config = cls.PRUNING_TIERS.get(tier, {})
        return tier_config.get("max_memories", 50000)
    
    @classmethod
    def should_consolidate(cls, category: MemoryCategory, similar_count: int) -> bool:
        """Check if memories should be consolidated based on category policy."""
        policy = cls.get_policy(category)
        return similar_count >= policy.consolidation_threshold
    
    @classmethod
    def calculate_decay_factor(cls, category: MemoryCategory, age_hours: float) -> float:
        """Calculate decay factor based on memory age and category."""
        policy = cls.get_policy(category)
        # Exponential decay: factor = e^(-decay_rate * age_hours)
        import math
        decay_factor = math.exp(-policy.decay_rate * age_hours)
        return max(0.1, decay_factor)  # Minimum 10% retention
    
    @classmethod
    def get_batch_size(cls, category: MemoryCategory) -> int:
        """Get optimal batch size for category operations."""
        policy = cls.get_policy(category)
        return policy.batch_size
    
    @classmethod
    def requires_graph_pruning(cls, category: MemoryCategory) -> bool:
        """Check if category requires graph-based pruning."""
        policy = cls.get_policy(category)
        return policy.requires_graph_pruning
    
    @classmethod
    def get_custom_filters(cls, category: MemoryCategory) -> Dict[str, Any]:
        """Get custom filters for category-specific queries."""
        policy = cls.get_policy(category)
        return policy.custom_filters or {}


class MemoryMetadata:
    """Helper class for memory metadata management."""
    
    @staticmethod
    def create_metadata(
        category: MemoryCategory,
        confidence: float,
        source: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Create standardized metadata for memory storage."""
        metadata = {
            "category": category.value,
            "confidence": confidence,
            "source": source,
            "created_at": kwargs.get("created_at"),
            "last_accessed": kwargs.get("last_accessed"),
            "access_count": kwargs.get("access_count", 0),
            "importance_score": kwargs.get("importance_score", confidence),
            "tags": kwargs.get("tags", []),
            "relationships": kwargs.get("relationships", []),
        }
        
        # Add category-specific metadata
        policy = MemoryCategoryConfig.get_policy(category)
        custom_filters = policy.custom_filters or {}
        
        for key, value_constraint in custom_filters.items():
            if key in kwargs:
                metadata[key] = kwargs[key]
        
        return {k: v for k, v in metadata.items() if v is not None}
    
    @staticmethod
    def update_access_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update metadata when memory is accessed."""
        from datetime import datetime
        
        updated = metadata.copy()
        updated["last_accessed"] = datetime.utcnow().isoformat()
        updated["access_count"] = updated.get("access_count", 0) + 1
        
        # Boost importance score based on access frequency
        access_count = updated["access_count"]
        confidence = updated.get("confidence", 0.5)
        
        # Logarithmic boost to prevent infinite growth
        import math
        access_boost = min(0.2, 0.05 * math.log(access_count + 1))
        updated["importance_score"] = min(1.0, confidence + access_boost)
        
        return updated