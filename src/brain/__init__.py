"""
Aerodrome Brain - Core Knowledge Base System

A comprehensive AI-powered brain system for the Aerodrome protocol that integrates:
- Confidence scoring and metrics tracking
- Protocol knowledge base with real-time data
- Natural language query processing
- AI-powered insights and analysis
- Memory management and pruning

This is the central intelligence system that coordinates all AI agent capabilities.
"""

from .confidence_scorer import (
    ConfidenceScorer,
    MemoryCategory,
    MemoryItem,
    DataSourceType,
    ConfidenceFactors,
    ConfidenceThresholds,
    ConfidenceScorerConfig,
)

from .confidence_metrics import (
    MetricsCollector,
    MetricsAnalyzer,
    MetricsReporter,
    AccuracyMetric,
    PredictionValidation,
    ConfidenceDistribution,
    FactorCorrelation,
    ConfidenceMetricsConfig,
    MetricType,
)

from .core import (
    AerodromeBrain,
    BrainConfig,
    SystemStatus,
    ComponentStatus,
    ComponentHealth,
    SystemMetrics,
    create_aerodrome_brain,
)

from .knowledge_base import (
    ProtocolKnowledgeBase,
    KnowledgeType,
    KnowledgeQuery,
    KnowledgeResponse,
    KnowledgeItem,
    DataFreshness,
    ProtocolStateManager,
    KnowledgeIndexer,
)

from .query_handler import (
    QueryHandler,
    QueryType,
    QueryIntent,
    QueryContext,
    QueryResponse,
    QueryAnalysis,
    ResponseFormat,
    QueryAnalyzer,
    ResponseGenerator,
    ConversationManager,
)

__version__ = "1.0.0"
__author__ = "Aerodrome AI Agent Team"

__all__ = [
    # Core brain orchestrator
    "AerodromeBrain",
    "BrainConfig",
    "SystemStatus",
    "ComponentStatus", 
    "ComponentHealth",
    "SystemMetrics",
    "create_aerodrome_brain",
    
    # Knowledge base
    "ProtocolKnowledgeBase",
    "KnowledgeType",
    "KnowledgeQuery",
    "KnowledgeResponse",
    "KnowledgeItem",
    "DataFreshness",
    "ProtocolStateManager",
    "KnowledgeIndexer",
    
    # Query processing
    "QueryHandler",
    "QueryType",
    "QueryIntent", 
    "QueryContext",
    "QueryResponse",
    "QueryAnalysis",
    "ResponseFormat",
    "QueryAnalyzer",
    "ResponseGenerator",
    "ConversationManager",
    
    # Core scoring
    "ConfidenceScorer",
    "MemoryCategory", 
    "MemoryItem",
    "DataSourceType",
    "ConfidenceFactors",
    "ConfidenceThresholds",
    "ConfidenceScorerConfig",
    
    # Metrics and analysis
    "MetricsCollector",
    "MetricsAnalyzer", 
    "MetricsReporter",
    "AccuracyMetric",
    "PredictionValidation",
    "ConfidenceDistribution",
    "FactorCorrelation",
    "ConfidenceMetricsConfig",
    "MetricType",
]