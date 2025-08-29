"""
Aerodrome Brain Core Orchestrator

Main orchestrator that integrates all components of the Aerodrome AI Agent:
- Confidence scoring and metrics
- Memory management with Mem0
- Protocol data monitoring
- AI intelligence and pattern recognition
- Query processing and response generation
- System health monitoring

This is the central coordination layer that manages data flow and system state.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

import structlog
from pydantic import BaseModel, Field

# Internal imports
from .confidence_scorer import (
    ConfidenceScorer,
    MemoryItem,
    MemoryCategory,
    DataSourceType,
    ConfidenceFactors,
    ConfidenceThresholds
)
from .confidence_metrics import MetricsCollector, MetricsAnalyzer, MetricsReporter
from .knowledge_base import ProtocolKnowledgeBase, KnowledgeQuery, KnowledgeResponse
from .query_handler import QueryHandler, QueryType, QueryContext, QueryResponse

from ..memory import EnhancedMem0Client, MemoryPruningEngine
from ..protocol import AerodromeClient, PoolMonitor, VotingAnalyzer
from ..intelligence import GeminiClient, PatternRecognitionEngine, PredictionEngine

logger = structlog.get_logger(__name__)


class SystemStatus(Enum):
    """Overall system status"""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class ComponentStatus(Enum):
    """Individual component status"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class ComponentHealth:
    """Health status for a system component"""
    name: str
    status: ComponentStatus
    last_update: datetime
    error_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time: float = 0.0
    memory_items: int = 0
    confidence_avg: float = 0.0
    uptime: timedelta = field(default_factory=timedelta)
    last_updated: datetime = field(default_factory=datetime.now)


class BrainConfig(BaseModel):
    """Configuration for the Brain core system"""
    
    # Component configurations
    confidence_config: Dict[str, Any] = Field(default_factory=dict)
    memory_config: Dict[str, Any] = Field(default_factory=dict)
    protocol_config: Dict[str, Any] = Field(default_factory=dict)
    intelligence_config: Dict[str, Any] = Field(default_factory=dict)
    
    # System settings
    health_check_interval: int = Field(default=30)  # seconds
    metrics_update_interval: int = Field(default=60)  # seconds
    auto_pruning_interval: int = Field(default=3600)  # seconds
    max_concurrent_queries: int = Field(default=10)
    
    # Logging configuration
    log_level: str = Field(default="INFO")
    enable_detailed_logging: bool = Field(default=True)
    
    # Safety settings
    max_memory_items: int = Field(default=100000)
    confidence_threshold: float = Field(default=0.3)
    emergency_shutdown_threshold: int = Field(default=100)  # consecutive errors


class AerodromeBrain:
    """
    Main orchestrator for the Aerodrome AI Agent brain system.
    
    Coordinates all components and manages system state, health monitoring,
    query processing, and data flow between subsystems.
    """
    
    def __init__(self, config: BrainConfig):
        """
        Initialize the brain orchestrator.
        
        Args:
            config: Brain configuration settings
        """
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # System state
        self.status = SystemStatus.STARTING
        self.start_time = datetime.now()
        self.component_health: Dict[str, ComponentHealth] = {}
        self.system_metrics = SystemMetrics()
        self.consecutive_errors = 0
        
        # Component references
        self.confidence_scorer: Optional[ConfidenceScorer] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.metrics_analyzer: Optional[MetricsAnalyzer] = None
        self.metrics_reporter: Optional[MetricsReporter] = None
        
        self.knowledge_base: Optional[ProtocolKnowledgeBase] = None
        self.query_handler: Optional[QueryHandler] = None
        
        self.memory_client: Optional[EnhancedMem0Client] = None
        self.pruning_engine: Optional[MemoryPruningEngine] = None
        
        self.aerodrome_client: Optional[AerodromeClient] = None
        self.pool_monitor: Optional[PoolMonitor] = None
        self.voting_analyzer: Optional[VotingAnalyzer] = None
        
        self.gemini_client: Optional[GeminiClient] = None
        self.pattern_engine: Optional[PatternRecognitionEngine] = None
        self.prediction_engine: Optional[PredictionEngine] = None
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._query_semaphore = asyncio.Semaphore(config.max_concurrent_queries)
        
        # Initialize component health tracking
        self._init_component_health()
    
    def _init_component_health(self) -> None:
        """Initialize component health tracking."""
        component_names = [
            "confidence_scorer", "metrics_collector", "knowledge_base",
            "query_handler", "memory_client", "aerodrome_client",
            "gemini_client", "pattern_engine", "prediction_engine"
        ]
        
        for name in component_names:
            self.component_health[name] = ComponentHealth(
                name=name,
                status=ComponentStatus.INITIALIZING,
                last_update=datetime.now()
            )
    
    async def initialize(self) -> None:
        """
        Initialize all brain components and start background tasks.
        """
        try:
            await self.logger.ainfo("Starting brain initialization")
            
            # Initialize core components
            await self._init_confidence_system()
            await self._init_memory_system()
            await self._init_protocol_system()
            await self._init_intelligence_system()
            await self._init_knowledge_system()
            await self._init_query_system()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.status = SystemStatus.HEALTHY
            await self.logger.ainfo("Brain initialization completed successfully")
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            await self.logger.aerror("Brain initialization failed", error=str(e))
            raise
    
    async def _init_confidence_system(self) -> None:
        """Initialize confidence scoring system."""
        try:
            self.confidence_scorer = ConfidenceScorer()
            self.metrics_collector = MetricsCollector()
            self.metrics_analyzer = MetricsAnalyzer()
            self.metrics_reporter = MetricsReporter()
            
            self._update_component_health("confidence_scorer", ComponentStatus.READY)
            self._update_component_health("metrics_collector", ComponentStatus.READY)
            
            await self.logger.ainfo("Confidence system initialized")
            
        except Exception as e:
            self._update_component_health("confidence_scorer", ComponentStatus.ERROR, str(e))
            await self.logger.aerror("Failed to initialize confidence system", error=str(e))
            raise
    
    async def _init_memory_system(self) -> None:
        """Initialize memory management system."""
        try:
            # Initialize Mem0 client with configuration
            mem0_config = self.config.memory_config
            self.memory_client = EnhancedMem0Client(
                api_key=mem0_config.get("api_key"),
                config=mem0_config.get("client_config", {}),
                enable_graph=mem0_config.get("enable_graph", True),
                neo4j_config=mem0_config.get("neo4j_config")
            )
            
            # Initialize pruning engine
            self.pruning_engine = MemoryPruningEngine(self.memory_client)
            
            self._update_component_health("memory_client", ComponentStatus.READY)
            
            await self.logger.ainfo("Memory system initialized")
            
        except Exception as e:
            self._update_component_health("memory_client", ComponentStatus.ERROR, str(e))
            await self.logger.aerror("Failed to initialize memory system", error=str(e))
            raise
    
    async def _init_protocol_system(self) -> None:
        """Initialize protocol integration system."""
        try:
            protocol_config = self.config.protocol_config
            
            # Initialize Aerodrome client
            self.aerodrome_client = AerodromeClient(
                quicknode_url=protocol_config.get("quicknode_url"),
                config=protocol_config.get("client_config", {})
            )
            
            # Initialize monitoring components
            self.pool_monitor = PoolMonitor(
                self.aerodrome_client,
                config=protocol_config.get("monitor_config", {})
            )
            
            self.voting_analyzer = VotingAnalyzer(
                self.aerodrome_client,
                config=protocol_config.get("voting_config", {})
            )
            
            self._update_component_health("aerodrome_client", ComponentStatus.READY)
            
            await self.logger.ainfo("Protocol system initialized")
            
        except Exception as e:
            self._update_component_health("aerodrome_client", ComponentStatus.ERROR, str(e))
            await self.logger.aerror("Failed to initialize protocol system", error=str(e))
            raise
    
    async def _init_intelligence_system(self) -> None:
        """Initialize AI intelligence system."""
        try:
            intelligence_config = self.config.intelligence_config
            
            # Initialize Gemini client
            self.gemini_client = GeminiClient(
                api_key=intelligence_config.get("gemini_api_key"),
                model=intelligence_config.get("model", "gemini-2.0-flash-001"),
                config=intelligence_config.get("client_config", {})
            )
            
            # Initialize AI engines
            self.pattern_engine = PatternRecognitionEngine(self.gemini_client)
            self.prediction_engine = PredictionEngine(self.gemini_client)
            
            self._update_component_health("gemini_client", ComponentStatus.READY)
            self._update_component_health("pattern_engine", ComponentStatus.READY)
            self._update_component_health("prediction_engine", ComponentStatus.READY)
            
            await self.logger.ainfo("Intelligence system initialized")
            
        except Exception as e:
            self._update_component_health("gemini_client", ComponentStatus.ERROR, str(e))
            await self.logger.aerror("Failed to initialize intelligence system", error=str(e))
            raise
    
    async def _init_knowledge_system(self) -> None:
        """Initialize knowledge base system."""
        try:
            self.knowledge_base = ProtocolKnowledgeBase(
                memory_client=self.memory_client,
                aerodrome_client=self.aerodrome_client,
                confidence_scorer=self.confidence_scorer,
                gemini_client=self.gemini_client
            )
            
            await self.knowledge_base.initialize()
            
            self._update_component_health("knowledge_base", ComponentStatus.READY)
            
            await self.logger.ainfo("Knowledge base system initialized")
            
        except Exception as e:
            self._update_component_health("knowledge_base", ComponentStatus.ERROR, str(e))
            await self.logger.aerror("Failed to initialize knowledge base", error=str(e))
            raise
    
    async def _init_query_system(self) -> None:
        """Initialize query processing system."""
        try:
            self.query_handler = QueryHandler(
                knowledge_base=self.knowledge_base,
                gemini_client=self.gemini_client,
                confidence_scorer=self.confidence_scorer
            )
            
            self._update_component_health("query_handler", ComponentStatus.READY)
            
            await self.logger.ainfo("Query system initialized")
            
        except Exception as e:
            self._update_component_health("query_handler", ComponentStatus.ERROR, str(e))
            await self.logger.aerror("Failed to initialize query system", error=str(e))
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        try:
            # Health monitoring task
            health_task = asyncio.create_task(self._health_monitor_loop())
            self._background_tasks.append(health_task)
            
            # Metrics collection task
            metrics_task = asyncio.create_task(self._metrics_update_loop())
            self._background_tasks.append(metrics_task)
            
            # Memory pruning task
            pruning_task = asyncio.create_task(self._memory_pruning_loop())
            self._background_tasks.append(pruning_task)
            
            # Protocol data sync task
            if self.knowledge_base:
                sync_task = asyncio.create_task(self._protocol_sync_loop())
                self._background_tasks.append(sync_task)
            
            await self.logger.ainfo(f"Started {len(self._background_tasks)} background tasks")
            
        except Exception as e:
            await self.logger.aerror("Failed to start background tasks", error=str(e))
            raise
    
    async def _health_monitor_loop(self) -> None:
        """Background task for component health monitoring."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_component_health()
                await self._update_system_status()
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                await self.logger.aerror("Error in health monitor loop", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _metrics_update_loop(self) -> None:
        """Background task for metrics collection and analysis."""
        while not self._shutdown_event.is_set():
            try:
                await self._collect_system_metrics()
                await self._analyze_performance()
                await asyncio.sleep(self.config.metrics_update_interval)
                
            except Exception as e:
                await self.logger.aerror("Error in metrics update loop", error=str(e))
                await asyncio.sleep(10)
    
    async def _memory_pruning_loop(self) -> None:
        """Background task for automatic memory pruning."""
        while not self._shutdown_event.is_set():
            try:
                if self.pruning_engine and self.system_metrics.memory_items > 1000:
                    await self.pruning_engine.run_pruning_cycle()
                    await self.logger.ainfo("Completed automatic memory pruning cycle")
                
                await asyncio.sleep(self.config.auto_pruning_interval)
                
            except Exception as e:
                await self.logger.aerror("Error in memory pruning loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _protocol_sync_loop(self) -> None:
        """Background task for protocol data synchronization."""
        while not self._shutdown_event.is_set():
            try:
                if self.knowledge_base:
                    await self.knowledge_base.sync_protocol_data()
                
                await asyncio.sleep(300)  # Sync every 5 minutes
                
            except Exception as e:
                await self.logger.aerror("Error in protocol sync loop", error=str(e))
                await asyncio.sleep(60)
    
    def _update_component_health(
        self, 
        component: str, 
        status: ComponentStatus,
        error: Optional[str] = None
    ) -> None:
        """Update component health status."""
        if component in self.component_health:
            health = self.component_health[component]
            health.status = status
            health.last_update = datetime.now()
            
            if error:
                health.error_count += 1
                health.last_error = error
            elif status == ComponentStatus.READY:
                health.error_count = 0
                health.last_error = None
    
    async def _check_component_health(self) -> None:
        """Check health of all components."""
        for name, health in self.component_health.items():
            try:
                component = getattr(self, name, None)
                if component is None:
                    health.status = ComponentStatus.OFFLINE
                    continue
                
                # Component-specific health checks
                if hasattr(component, 'health_check'):
                    is_healthy = await component.health_check()
                    if is_healthy:
                        if health.status == ComponentStatus.ERROR:
                            health.status = ComponentStatus.READY
                    else:
                        health.status = ComponentStatus.ERROR
                
            except Exception as e:
                health.status = ComponentStatus.ERROR
                health.error_count += 1
                health.last_error = str(e)
    
    async def _update_system_status(self) -> None:
        """Update overall system status based on component health."""
        error_components = sum(
            1 for h in self.component_health.values() 
            if h.status == ComponentStatus.ERROR
        )
        
        offline_components = sum(
            1 for h in self.component_health.values()
            if h.status == ComponentStatus.OFFLINE
        )
        
        total_components = len(self.component_health)
        
        if error_components > total_components // 2:
            self.status = SystemStatus.ERROR
        elif error_components > 0 or offline_components > 0:
            self.status = SystemStatus.DEGRADED
        else:
            self.status = SystemStatus.HEALTHY
        
        # Check for emergency shutdown
        if self.consecutive_errors >= self.config.emergency_shutdown_threshold:
            await self.logger.aerror(
                f"Emergency shutdown triggered after {self.consecutive_errors} consecutive errors"
            )
            await self.shutdown()
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-wide metrics."""
        try:
            # Update uptime
            self.system_metrics.uptime = datetime.now() - self.start_time
            
            # Collect memory metrics
            if self.memory_client:
                try:
                    # This would need to be implemented in the memory client
                    memory_stats = getattr(self.memory_client, 'get_stats', lambda: {})()
                    self.system_metrics.memory_items = memory_stats.get('total_items', 0)
                except:
                    pass
            
            # Collect confidence metrics
            if self.confidence_scorer and hasattr(self.confidence_scorer, 'get_average_confidence'):
                try:
                    self.system_metrics.confidence_avg = await self.confidence_scorer.get_average_confidence()
                except:
                    pass
            
            self.system_metrics.last_updated = datetime.now()
            
        except Exception as e:
            await self.logger.aerror("Error collecting system metrics", error=str(e))
    
    async def _analyze_performance(self) -> None:
        """Analyze system performance and adjust parameters."""
        try:
            if self.metrics_analyzer:
                # Analyze recent performance
                performance_data = await self.metrics_analyzer.analyze_recent_performance()
                
                # Adjust confidence scorer weights if needed
                if self.confidence_scorer and "factor_performance" in performance_data:
                    await self.confidence_scorer.adjust_factor_weights(
                        performance_data["factor_performance"]
                    )
                
        except Exception as e:
            await self.logger.aerror("Error analyzing performance", error=str(e))
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> QueryResponse:
        """
        Process a user query through the brain system.
        
        Args:
            query: Natural language query
            context: Optional context information
            user_id: Optional user identifier
            
        Returns:
            QueryResponse with answer and metadata
        """
        start_time = time.time()
        
        async with self._query_semaphore:
            try:
                if not self.query_handler:
                    raise RuntimeError("Query handler not initialized")
                
                # Track query
                self.system_metrics.total_queries += 1
                
                # Create query context
                query_context = QueryContext(
                    query=query,
                    user_id=user_id,
                    timestamp=datetime.now(),
                    context=context or {}
                )
                
                # Process query
                response = await self.query_handler.process_query(query_context)
                
                # Update metrics
                response_time = time.time() - start_time
                self.system_metrics.successful_queries += 1
                self._update_response_time(response_time)
                
                # Reset consecutive error count on success
                self.consecutive_errors = 0
                
                await self.logger.ainfo(
                    "Query processed successfully",
                    query_id=response.query_id,
                    response_time=response_time,
                    confidence=response.confidence
                )
                
                return response
                
            except Exception as e:
                self.system_metrics.failed_queries += 1
                self.consecutive_errors += 1
                
                await self.logger.aerror(
                    "Query processing failed",
                    query=query[:100],  # Log first 100 chars
                    error=str(e),
                    user_id=user_id
                )
                
                # Return error response
                return QueryResponse(
                    query_id=f"error_{int(time.time())}",
                    response="I apologize, but I encountered an error processing your query. Please try again.",
                    confidence=0.0,
                    sources=[],
                    metadata={
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                )
    
    def _update_response_time(self, response_time: float) -> None:
        """Update average response time metric."""
        if self.system_metrics.successful_queries <= 1:
            self.system_metrics.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.system_metrics.avg_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.system_metrics.avg_response_time
            )
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health and metrics."""
        return {
            "system_status": self.status.value,
            "uptime": str(self.system_metrics.uptime),
            "component_health": {
                name: {
                    "status": health.status.value,
                    "last_update": health.last_update.isoformat(),
                    "error_count": health.error_count,
                    "last_error": health.last_error
                }
                for name, health in self.component_health.items()
            },
            "metrics": {
                "total_queries": self.system_metrics.total_queries,
                "successful_queries": self.system_metrics.successful_queries,
                "failed_queries": self.system_metrics.failed_queries,
                "success_rate": (
                    self.system_metrics.successful_queries / 
                    max(self.system_metrics.total_queries, 1)
                ),
                "avg_response_time": self.system_metrics.avg_response_time,
                "memory_items": self.system_metrics.memory_items,
                "confidence_avg": self.system_metrics.confidence_avg
            }
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the brain system."""
        try:
            self.status = SystemStatus.SHUTTING_DOWN
            await self.logger.ainfo("Starting brain shutdown")
            
            # Signal shutdown to background tasks
            self._shutdown_event.set()
            
            # Wait for background tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Shutdown components
            components_to_shutdown = [
                self.memory_client,
                self.aerodrome_client,
                self.gemini_client
            ]
            
            for component in components_to_shutdown:
                if component and hasattr(component, 'shutdown'):
                    try:
                        await component.shutdown()
                    except Exception as e:
                        await self.logger.awarning(
                            "Error shutting down component", 
                            component=type(component).__name__,
                            error=str(e)
                        )
            
            await self.logger.ainfo("Brain shutdown completed")
            
        except Exception as e:
            await self.logger.aerror("Error during brain shutdown", error=str(e))
    
    @asynccontextmanager
    async def lifespan(self):
        """Async context manager for brain lifecycle management."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()


# Factory function for easier initialization
async def create_aerodrome_brain(config: BrainConfig) -> AerodromeBrain:
    """
    Factory function to create and initialize an Aerodrome Brain instance.
    
    Args:
        config: Brain configuration
        
    Returns:
        Initialized AerodromeBrain instance
    """
    brain = AerodromeBrain(config)
    await brain.initialize()
    return brain


# Example usage
async def main():
    """Example usage of the Aerodrome Brain."""
    
    # Configure the brain
    config = BrainConfig(
        memory_config={
            "api_key": "your_mem0_api_key",
            "enable_graph": True
        },
        protocol_config={
            "quicknode_url": "your_quicknode_url"
        },
        intelligence_config={
            "gemini_api_key": "your_gemini_api_key"
        },
        health_check_interval=30,
        max_concurrent_queries=5
    )
    
    # Create and use the brain
    async with AerodromeBrain(config).lifespan() as brain:
        # Process some queries
        response1 = await brain.process_query(
            "What are the top performing pools on Aerodrome today?"
        )
        print(f"Response 1: {response1.response}")
        
        response2 = await brain.process_query(
            "How does veAERO voting work?",
            context={"user_level": "beginner"}
        )
        print(f"Response 2: {response2.response}")
        
        # Check system health
        health = await brain.get_system_health()
        print(f"System health: {health['system_status']}")


if __name__ == "__main__":
    asyncio.run(main())