"""
Comprehensive confidence scoring system for the Aerodrome brain.

This module provides a multi-factor confidence calculation system that evaluates
data source reliability, historical accuracy, recency, corroboration, and sample size
to generate dynamic confidence scores with decay over time.
"""

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog
from pydantic import BaseModel, Field, validator

logger = structlog.get_logger(__name__)


class MemoryCategory(Enum):
    """Memory categories with different confidence thresholds."""
    
    PROTOCOL_CONSTANTS = "protocol_constants"
    POOL_PERFORMANCE = "pool_performance"
    VOTING_PATTERNS = "voting_patterns"
    MARKET_CORRELATIONS = "market_correlations"
    SPECULATIVE_INSIGHTS = "speculative_insights"


class DataSourceType(Enum):
    """Types of data sources with different reliability scores."""
    
    BLOCKCHAIN_DATA = "blockchain_data"  # Highest reliability
    API_ENDPOINT = "api_endpoint"
    USER_INPUT = "user_input"
    PREDICTED_VALUE = "predicted_value"  # Lowest reliability


@dataclass
class ConfidenceFactors:
    """Factors used in confidence calculation."""
    
    data_source_reliability: float = 0.0
    historical_accuracy: float = 0.0
    recency: float = 0.0
    corroboration: float = 0.0
    sample_size: float = 0.0
    prediction_success_rate: float = 0.0


@dataclass
class MemoryItem:
    """Represents a memory item with confidence tracking."""
    
    id: str
    category: MemoryCategory
    data: Dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime
    source_type: DataSourceType
    prediction_outcomes: List[bool] = field(default_factory=list)
    corroborating_sources: Set[str] = field(default_factory=set)
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class ConfidenceThresholds:
    """Confidence thresholds for different memory categories."""
    
    THRESHOLDS = {
        MemoryCategory.PROTOCOL_CONSTANTS: 0.95,
        MemoryCategory.POOL_PERFORMANCE: 0.70,
        MemoryCategory.VOTING_PATTERNS: 0.60,
        MemoryCategory.MARKET_CORRELATIONS: 0.50,
        MemoryCategory.SPECULATIVE_INSIGHTS: 0.30,
    }
    
    RETENTION_PERIODS = {
        MemoryCategory.PROTOCOL_CONSTANTS: timedelta(days=365),  # Permanent
        MemoryCategory.POOL_PERFORMANCE: timedelta(days=30),
        MemoryCategory.VOTING_PATTERNS: timedelta(days=14),
        MemoryCategory.MARKET_CORRELATIONS: timedelta(days=7),
        MemoryCategory.SPECULATIVE_INSIGHTS: timedelta(days=2),
    }
    
    @classmethod
    def get_threshold(cls, category: MemoryCategory) -> float:
        """Get confidence threshold for a category."""
        return cls.THRESHOLDS.get(category, 0.5)
    
    @classmethod
    def get_retention_period(cls, category: MemoryCategory) -> timedelta:
        """Get retention period for a category."""
        return cls.RETENTION_PERIODS.get(category, timedelta(days=7))


class DataSourceReliabilityScorer:
    """Scores data source reliability."""
    
    RELIABILITY_SCORES = {
        DataSourceType.BLOCKCHAIN_DATA: 1.0,
        DataSourceType.API_ENDPOINT: 0.8,
        DataSourceType.USER_INPUT: 0.6,
        DataSourceType.PREDICTED_VALUE: 0.4,
    }
    
    @classmethod
    def score(cls, source_type: DataSourceType, metadata: Optional[Dict] = None) -> float:
        """Calculate reliability score for a data source."""
        base_score = cls.RELIABILITY_SCORES.get(source_type, 0.5)
        
        # Adjust based on metadata
        if metadata:
            # Adjust for API endpoint reputation
            if source_type == DataSourceType.API_ENDPOINT:
                if metadata.get("provider") == "quicknode":
                    base_score += 0.1
                elif metadata.get("uptime", 0) > 0.99:
                    base_score += 0.05
            
            # Adjust for prediction model confidence
            if source_type == DataSourceType.PREDICTED_VALUE:
                model_confidence = metadata.get("model_confidence", 0.5)
                base_score = base_score * model_confidence
        
        return min(base_score, 1.0)


class RecencyScorer:
    """Scores data recency using exponential decay."""
    
    @staticmethod
    def score(
        created_at: datetime,
        category: MemoryCategory,
        decay_factor: float = 0.1
    ) -> float:
        """
        Calculate recency score with exponential decay.
        
        Args:
            created_at: When the data was created
            category: Memory category for context-specific decay
            decay_factor: Rate of decay (higher = faster decay)
            
        Returns:
            Recency score between 0 and 1
        """
        now = datetime.now()
        age_hours = (now - created_at).total_seconds() / 3600
        
        # Category-specific decay rates
        category_decay_factors = {
            MemoryCategory.PROTOCOL_CONSTANTS: 0.001,  # Very slow decay
            MemoryCategory.POOL_PERFORMANCE: 0.01,
            MemoryCategory.VOTING_PATTERNS: 0.02,
            MemoryCategory.MARKET_CORRELATIONS: 0.05,
            MemoryCategory.SPECULATIVE_INSIGHTS: 0.1,  # Fast decay
        }
        
        effective_decay = category_decay_factors.get(category, decay_factor)
        return math.exp(-effective_decay * age_hours)


class CorroborationScorer:
    """Scores data corroboration from multiple sources."""
    
    @staticmethod
    def score(
        corroborating_sources: Set[str],
        source_reliability_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate corroboration score based on number and quality of sources.
        
        Args:
            corroborating_sources: Set of source identifiers
            source_reliability_scores: Optional reliability scores for sources
            
        Returns:
            Corroboration score between 0 and 1
        """
        if not corroborating_sources:
            return 0.0
        
        num_sources = len(corroborating_sources)
        
        # Base score from number of sources (diminishing returns)
        base_score = 1 - math.exp(-0.5 * num_sources)
        
        # Weight by source reliability if available
        if source_reliability_scores:
            total_reliability = sum(
                source_reliability_scores.get(source, 0.5)
                for source in corroborating_sources
            )
            avg_reliability = total_reliability / num_sources
            base_score *= avg_reliability
        
        return min(base_score, 1.0)


class SampleSizeScorer:
    """Scores confidence based on sample size."""
    
    @staticmethod
    def score(sample_size: int, category: MemoryCategory) -> float:
        """
        Calculate confidence based on sample size using log scale.
        
        Args:
            sample_size: Number of data points
            category: Memory category for context-specific requirements
            
        Returns:
            Sample size score between 0 and 1
        """
        if sample_size <= 0:
            return 0.0
        
        # Category-specific minimum sample sizes
        min_samples = {
            MemoryCategory.PROTOCOL_CONSTANTS: 1,
            MemoryCategory.POOL_PERFORMANCE: 10,
            MemoryCategory.VOTING_PATTERNS: 5,
            MemoryCategory.MARKET_CORRELATIONS: 20,
            MemoryCategory.SPECULATIVE_INSIGHTS: 3,
        }
        
        min_required = min_samples.get(category, 5)
        
        # Use log scale for diminishing returns
        if sample_size >= min_required:
            # Full confidence at minimum, diminishing returns after
            score = 1.0 - math.exp(-0.1 * (sample_size - min_required + 1))
        else:
            # Linear penalty for insufficient samples
            score = sample_size / min_required
        
        return min(score, 1.0)


class HistoricalAccuracyTracker:
    """Tracks and scores historical accuracy of predictions."""
    
    def __init__(self):
        self.prediction_history: Dict[str, List[bool]] = {}
        self.accuracy_cache: Dict[str, float] = {}
        self.cache_expiry: Dict[str, datetime] = {}
    
    def add_prediction_outcome(self, memory_id: str, was_accurate: bool) -> None:
        """Add a prediction outcome for a memory item."""
        if memory_id not in self.prediction_history:
            self.prediction_history[memory_id] = []
        
        self.prediction_history[memory_id].append(was_accurate)
        
        # Keep only recent predictions (last 100)
        if len(self.prediction_history[memory_id]) > 100:
            self.prediction_history[memory_id] = self.prediction_history[memory_id][-100:]
        
        # Invalidate cache
        if memory_id in self.accuracy_cache:
            del self.accuracy_cache[memory_id]
            del self.cache_expiry[memory_id]
    
    def get_accuracy_score(self, memory_id: str) -> float:
        """Get historical accuracy score for a memory item."""
        # Check cache
        if (memory_id in self.accuracy_cache and 
            memory_id in self.cache_expiry and
            datetime.now() < self.cache_expiry[memory_id]):
            return self.accuracy_cache[memory_id]
        
        history = self.prediction_history.get(memory_id, [])
        if not history:
            return 0.5  # Neutral score for no history
        
        # Calculate weighted accuracy (recent predictions weighted more)
        weights = [math.exp(-0.01 * i) for i in range(len(history))]
        weighted_accuracy = sum(
            outcome * weight for outcome, weight in zip(history, reversed(weights))
        ) / sum(weights)
        
        # Cache the result
        self.accuracy_cache[memory_id] = weighted_accuracy
        self.cache_expiry[memory_id] = datetime.now() + timedelta(hours=1)
        
        return weighted_accuracy


class ConfidenceDecayCalculator:
    """Calculates confidence decay over time."""
    
    @staticmethod
    def apply_decay(
        initial_confidence: float,
        created_at: datetime,
        category: MemoryCategory,
        access_pattern: Optional[Dict] = None
    ) -> float:
        """
        Apply time-based confidence decay.
        
        Args:
            initial_confidence: Original confidence score
            created_at: When the memory was created
            category: Memory category
            access_pattern: Optional access pattern data
            
        Returns:
            Decayed confidence score
        """
        age = datetime.now() - created_at
        age_hours = age.total_seconds() / 3600
        
        # Category-specific decay rates
        decay_rates = {
            MemoryCategory.PROTOCOL_CONSTANTS: 0.0001,  # Very slow decay
            MemoryCategory.POOL_PERFORMANCE: 0.001,
            MemoryCategory.VOTING_PATTERNS: 0.002,
            MemoryCategory.MARKET_CORRELATIONS: 0.005,
            MemoryCategory.SPECULATIVE_INSIGHTS: 0.01,  # Fast decay
        }
        
        decay_rate = decay_rates.get(category, 0.002)
        
        # Adjust decay based on access pattern
        if access_pattern:
            access_count = access_pattern.get("access_count", 0)
            last_accessed = access_pattern.get("last_accessed")
            
            if last_accessed and access_count > 0:
                # Slow decay for frequently accessed items
                hours_since_access = (datetime.now() - last_accessed).total_seconds() / 3600
                access_factor = min(1.0, access_count / 10.0)  # Cap at 10 accesses
                decay_rate *= (1 - 0.5 * access_factor * math.exp(-0.01 * hours_since_access))
        
        # Apply exponential decay
        decayed_confidence = initial_confidence * math.exp(-decay_rate * age_hours)
        
        return max(decayed_confidence, 0.0)


class ConfidenceScorer:
    """Main confidence scoring engine with multi-factor calculation."""
    
    def __init__(self):
        self.accuracy_tracker = HistoricalAccuracyTracker()
        self.decay_calculator = ConfidenceDecayCalculator()
        self.logger = structlog.get_logger(__name__)
        
        # Configurable factor weights
        self.factor_weights = {
            "data_source_reliability": 0.25,
            "historical_accuracy": 0.20,
            "recency": 0.20,
            "corroboration": 0.15,
            "sample_size": 0.15,
            "prediction_success_rate": 0.05,
        }
    
    async def calculate_confidence(
        self,
        memory_item: MemoryItem,
        metadata: Optional[Dict] = None
    ) -> Tuple[float, ConfidenceFactors]:
        """
        Calculate comprehensive confidence score for a memory item.
        
        Args:
            memory_item: The memory item to score
            metadata: Optional metadata for scoring context
            
        Returns:
            Tuple of (confidence_score, confidence_factors)
        """
        try:
            # Calculate individual factor scores
            factors = ConfidenceFactors()
            
            # Data source reliability
            factors.data_source_reliability = DataSourceReliabilityScorer.score(
                memory_item.source_type, metadata
            )
            
            # Historical accuracy
            factors.historical_accuracy = self.accuracy_tracker.get_accuracy_score(
                memory_item.id
            )
            
            # Recency score
            factors.recency = RecencyScorer.score(
                memory_item.created_at, memory_item.category
            )
            
            # Corroboration score
            source_reliability_map = {}
            if metadata and "source_reliabilities" in metadata:
                source_reliability_map = metadata["source_reliabilities"]
            
            factors.corroboration = CorroborationScorer.score(
                memory_item.corroborating_sources, source_reliability_map
            )
            
            # Sample size score
            sample_size = 1
            if metadata and "sample_size" in metadata:
                sample_size = metadata["sample_size"]
            elif "sample_size" in memory_item.data:
                sample_size = memory_item.data["sample_size"]
            
            factors.sample_size = SampleSizeScorer.score(
                sample_size, memory_item.category
            )
            
            # Prediction success rate
            if memory_item.prediction_outcomes:
                factors.prediction_success_rate = sum(memory_item.prediction_outcomes) / len(
                    memory_item.prediction_outcomes
                )
            else:
                factors.prediction_success_rate = 0.5  # Neutral for no predictions
            
            # Calculate weighted confidence score
            confidence_score = (
                factors.data_source_reliability * self.factor_weights["data_source_reliability"] +
                factors.historical_accuracy * self.factor_weights["historical_accuracy"] +
                factors.recency * self.factor_weights["recency"] +
                factors.corroboration * self.factor_weights["corroboration"] +
                factors.sample_size * self.factor_weights["sample_size"] +
                factors.prediction_success_rate * self.factor_weights["prediction_success_rate"]
            )
            
            # Apply confidence decay
            access_pattern = {
                "access_count": memory_item.access_count,
                "last_accessed": memory_item.last_accessed
            }
            
            confidence_score = self.decay_calculator.apply_decay(
                confidence_score,
                memory_item.created_at,
                memory_item.category,
                access_pattern
            )
            
            # Ensure score is within bounds
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            await self.logger.adebug(
                "Calculated confidence score",
                memory_id=memory_item.id,
                category=memory_item.category.value,
                confidence=confidence_score,
                factors=factors
            )
            
            return confidence_score, factors
            
        except Exception as e:
            await self.logger.aerror(
                "Error calculating confidence score",
                memory_id=memory_item.id,
                error=str(e)
            )
            # Return default confidence score on error
            return 0.5, ConfidenceFactors()
    
    async def update_prediction_outcome(
        self, 
        memory_id: str, 
        was_accurate: bool,
        memory_item: Optional[MemoryItem] = None
    ) -> None:
        """
        Update prediction outcome and recalculate confidence.
        
        Args:
            memory_id: ID of the memory item
            was_accurate: Whether the prediction was accurate
            memory_item: Optional memory item for recalculation
        """
        try:
            # Update accuracy tracker
            self.accuracy_tracker.add_prediction_outcome(memory_id, was_accurate)
            
            # Update memory item if provided
            if memory_item:
                memory_item.prediction_outcomes.append(was_accurate)
                memory_item.updated_at = datetime.now()
                
                # Recalculate confidence
                new_confidence, factors = await self.calculate_confidence(memory_item)
                memory_item.confidence = new_confidence
            
            await self.logger.ainfo(
                "Updated prediction outcome",
                memory_id=memory_id,
                was_accurate=was_accurate,
                new_confidence=memory_item.confidence if memory_item else None
            )
            
        except Exception as e:
            await self.logger.aerror(
                "Error updating prediction outcome",
                memory_id=memory_id,
                error=str(e)
            )
    
    def should_retain_memory(self, memory_item: MemoryItem) -> bool:
        """
        Determine if a memory item should be retained based on confidence and age.
        
        Args:
            memory_item: Memory item to evaluate
            
        Returns:
            True if the memory should be retained
        """
        # Check confidence threshold
        threshold = ConfidenceThresholds.get_threshold(memory_item.category)
        if memory_item.confidence < threshold:
            return False
        
        # Check retention period
        retention_period = ConfidenceThresholds.get_retention_period(memory_item.category)
        age = datetime.now() - memory_item.created_at
        
        if age > retention_period:
            # Allow retention if confidence is very high
            if memory_item.confidence < 0.9:
                return False
        
        return True
    
    async def adjust_factor_weights(
        self, 
        performance_metrics: Dict[str, float]
    ) -> None:
        """
        Dynamically adjust factor weights based on system performance.
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        try:
            # Adjust weights based on which factors correlate with accuracy
            if "accuracy_by_factor" in performance_metrics:
                factor_accuracies = performance_metrics["accuracy_by_factor"]
                
                # Increase weights for factors that correlate with accuracy
                total_adjustment = 0.0
                for factor, accuracy in factor_accuracies.items():
                    if factor in self.factor_weights:
                        adjustment = (accuracy - 0.5) * 0.1  # Max 5% adjustment
                        self.factor_weights[factor] += adjustment
                        total_adjustment += adjustment
                
                # Normalize weights to sum to 1.0
                total_weight = sum(self.factor_weights.values())
                if total_weight > 0:
                    for factor in self.factor_weights:
                        self.factor_weights[factor] /= total_weight
            
            await self.logger.ainfo(
                "Adjusted factor weights",
                new_weights=self.factor_weights,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            await self.logger.aerror(
                "Error adjusting factor weights",
                error=str(e)
            )


# Example usage and configuration
class ConfidenceScorerConfig(BaseModel):
    """Configuration for confidence scorer."""
    
    factor_weights: Dict[str, float] = Field(
        default={
            "data_source_reliability": 0.25,
            "historical_accuracy": 0.20,
            "recency": 0.20,
            "corroboration": 0.15,
            "sample_size": 0.15,
            "prediction_success_rate": 0.05,
        }
    )
    
    cache_expiry_hours: int = Field(default=1)
    max_prediction_history: int = Field(default=100)
    
    @validator("factor_weights")
    def weights_sum_to_one(cls, v):
        """Ensure factor weights sum to approximately 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Factor weights must sum to 1.0, got {total}")
        return v


async def main():
    """Example usage of the confidence scoring system."""
    # Initialize scorer
    scorer = ConfidenceScorer()
    
    # Create a sample memory item
    memory_item = MemoryItem(
        id="test_pool_performance",
        category=MemoryCategory.POOL_PERFORMANCE,
        data={
            "pool_address": "0x123...",
            "volume_24h": 1000000,
            "fees_24h": 3000,
            "sample_size": 144  # 24 hours of 10-minute samples
        },
        confidence=0.0,  # Will be calculated
        created_at=datetime.now() - timedelta(hours=2),
        updated_at=datetime.now(),
        source_type=DataSourceType.API_ENDPOINT,
        corroborating_sources={"quicknode", "dexscreener"}
    )
    
    # Calculate confidence
    confidence, factors = await scorer.calculate_confidence(
        memory_item,
        metadata={
            "provider": "quicknode",
            "uptime": 0.999,
            "sample_size": 144
        }
    )
    
    print(f"Calculated confidence: {confidence:.3f}")
    print(f"Factors: {factors}")
    
    # Update with prediction outcome
    await scorer.update_prediction_outcome("test_pool_performance", True, memory_item)
    
    # Check if memory should be retained
    should_retain = scorer.should_retain_memory(memory_item)
    print(f"Should retain memory: {should_retain}")


if __name__ == "__main__":
    asyncio.run(main())