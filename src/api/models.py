"""
Pydantic models for the Aerodrome AI Agent API

Request/response models, validation schemas, and error models
for the FastAPI application.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


# Enums
class SystemStatus(str, Enum):
    """System status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    ERROR = "error"
    STARTING = "starting"
    SHUTTING_DOWN = "shutting_down"


class ComponentStatus(str, Enum):
    """Component status enumeration"""
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


class QueryType(str, Enum):
    """Query type enumeration"""
    GENERAL = "general"
    PROTOCOL_INFO = "protocol_info"
    POOL_ANALYSIS = "pool_analysis"
    VOTING_INFO = "voting_info"
    PRICE_ANALYSIS = "price_analysis"
    YIELD_ANALYSIS = "yield_analysis"


class MessageType(str, Enum):
    """WebSocket message types"""
    PING = "ping"
    PONG = "pong"
    QUERY = "query"
    QUERY_RESULT = "query_result"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    WELCOME = "welcome"
    NOTIFICATION = "notification"


# Base Models
class BaseAPIModel(BaseModel):
    """Base model with common configuration"""
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
        use_enum_values = True
        validate_assignment = True


class TimestampedModel(BaseAPIModel):
    """Base model with timestamp"""
    timestamp: datetime = Field(default_factory=datetime.now)


# Request Models
class QueryRequest(BaseAPIModel):
    """Request model for query processing"""
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="Natural language query about Aerodrome Protocol"
    )
    user_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Optional user identifier for personalization"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional context information to help with query processing"
    )
    query_type: Optional[QueryType] = Field(
        QueryType.GENERAL,
        description="Type of query for optimized processing"
    )
    include_sources: bool = Field(
        True,
        description="Whether to include source information in response"
    )
    max_response_length: Optional[int] = Field(
        None,
        ge=100,
        le=5000,
        description="Maximum length of response in characters"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query content"""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @validator('context')
    def validate_context(cls, v):
        """Validate context size"""
        if v is not None:
            # Limit context size to prevent abuse
            serialized = str(v)
            if len(serialized) > 10000:
                raise ValueError("Context is too large")
        return v


class WebSocketMessage(BaseAPIModel):
    """WebSocket message model"""
    
    type: MessageType = Field(..., description="Message type")
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Message payload data"
    )
    message_id: Optional[str] = Field(
        None,
        description="Optional message ID for tracking"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Message timestamp"
    )


# Response Models
class SourceInfo(BaseAPIModel):
    """Information about a response source"""
    
    type: str = Field(..., description="Type of source (e.g., 'protocol_data', 'memory', 'knowledge_base')")
    title: Optional[str] = Field(None, description="Human-readable title of the source")
    url: Optional[str] = Field(None, description="URL if source is web-based")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this source")
    timestamp: Optional[datetime] = Field(None, description="When this source was last updated")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional source metadata")


class QueryResponse(TimestampedModel):
    """Response model for query processing"""
    
    query_id: str = Field(..., description="Unique identifier for this query")
    query: str = Field(..., description="Original query text")
    response: str = Field(..., description="AI-generated response")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score for the response"
    )
    sources: List[SourceInfo] = Field(
        default_factory=list,
        description="Sources used to generate the response"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the query processing"
    )
    processing_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Time taken to process the query in seconds"
    )
    suggested_followups: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )
    
    @validator('response')
    def validate_response(cls, v):
        """Validate response content"""
        if not v.strip():
            raise ValueError("Response cannot be empty")
        return v.strip()


class ComponentHealth(BaseAPIModel):
    """Health status of a system component"""
    
    name: str = Field(..., description="Component name")
    status: ComponentStatus = Field(..., description="Current status")
    last_update: datetime = Field(..., description="Last update timestamp")
    error_count: int = Field(0, ge=0, description="Number of errors encountered")
    last_error: Optional[str] = Field(None, description="Last error message")
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Component-specific metrics"
    )


class SystemMetrics(BaseAPIModel):
    """System performance metrics"""
    
    total_queries: int = Field(0, ge=0, description="Total number of queries processed")
    successful_queries: int = Field(0, ge=0, description="Number of successful queries")
    failed_queries: int = Field(0, ge=0, description="Number of failed queries")
    success_rate: float = Field(0.0, ge=0.0, le=1.0, description="Query success rate")
    avg_response_time: float = Field(0.0, ge=0.0, description="Average response time in seconds")
    memory_items: int = Field(0, ge=0, description="Number of items in memory")
    confidence_avg: float = Field(0.0, ge=0.0, le=1.0, description="Average confidence score")
    uptime: Optional[str] = Field(None, description="System uptime")
    
    @validator('success_rate')
    def calculate_success_rate(cls, v, values):
        """Calculate success rate from query counts"""
        total = values.get('total_queries', 0)
        successful = values.get('successful_queries', 0)
        return successful / max(total, 1)


class HealthResponse(TimestampedModel):
    """Health check response model"""
    
    status: SystemStatus = Field(..., description="Overall system status")
    version: str = Field(..., description="API version")
    uptime: Optional[str] = Field(None, description="System uptime")
    components: Optional[Dict[str, ComponentHealth]] = Field(
        None,
        description="Health status of individual components"
    )
    error: Optional[str] = Field(None, description="Error message if status is ERROR")


class MetricsResponse(TimestampedModel):
    """Detailed metrics response model"""
    
    system_metrics: SystemMetrics = Field(..., description="System-wide metrics")
    component_health: Dict[str, ComponentHealth] = Field(
        ...,
        description="Health status of all components"
    )
    api_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="API-specific metrics"
    )


class ErrorResponse(TimestampedModel):
    """Error response model"""
    
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    error_type: Optional[str] = Field(None, description="Error type classification")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    trace_id: Optional[str] = Field(None, description="Request trace ID for debugging")


# Specialized Response Models
class ProtocolInfoResponse(QueryResponse):
    """Response model for protocol information queries"""
    
    protocol_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured protocol data"
    )
    live_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Live protocol metrics"
    )


class PoolAnalysisResponse(QueryResponse):
    """Response model for pool analysis queries"""
    
    pools: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pool data and analysis"
    )
    market_trends: Optional[Dict[str, Any]] = Field(
        None,
        description="Market trend analysis"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Investment recommendations"
    )


class VotingInfoResponse(QueryResponse):
    """Response model for voting information queries"""
    
    active_votes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Currently active votes"
    )
    voting_power: Optional[Dict[str, float]] = Field(
        None,
        description="User's voting power breakdown"
    )
    vote_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent voting history"
    )


# WebSocket Response Models
class WebSocketQueryResult(BaseAPIModel):
    """WebSocket query result message"""
    
    type: MessageType = Field(MessageType.QUERY_RESULT, description="Message type")
    query_id: str = Field(..., description="Query ID")
    response: str = Field(..., description="AI response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
    sources: List[SourceInfo] = Field(default_factory=list, description="Response sources")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class WebSocketError(BaseAPIModel):
    """WebSocket error message"""
    
    type: MessageType = Field(MessageType.ERROR, description="Message type")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class WebSocketNotification(BaseAPIModel):
    """WebSocket notification message"""
    
    type: MessageType = Field(MessageType.NOTIFICATION, description="Message type")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    priority: str = Field("info", description="Notification priority (info, warning, error)")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional notification data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Notification timestamp")


# Request Validation Models
class QueryValidation(BaseAPIModel):
    """Query validation helper"""
    
    is_valid: bool = Field(..., description="Whether query is valid")
    issues: List[str] = Field(default_factory=list, description="Validation issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class RateLimitInfo(BaseAPIModel):
    """Rate limiting information"""
    
    limit: int = Field(..., description="Rate limit threshold")
    remaining: int = Field(..., description="Remaining requests")
    reset_time: datetime = Field(..., description="When the rate limit resets")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retrying")


# Admin Models
class BrainRestartRequest(BaseAPIModel):
    """Request model for brain restart"""
    
    reason: Optional[str] = Field(None, description="Reason for restart")
    force: bool = Field(False, description="Whether to force restart even if unhealthy")


class BrainRestartResponse(TimestampedModel):
    """Response model for brain restart"""
    
    status: str = Field(..., description="Restart status")
    message: str = Field(..., description="Status message")
    previous_uptime: Optional[str] = Field(None, description="Previous system uptime")


class MemoryPruningRequest(BaseAPIModel):
    """Request model for memory pruning"""
    
    aggressive: bool = Field(False, description="Whether to use aggressive pruning")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence threshold for pruning")


class MemoryPruningResponse(TimestampedModel):
    """Response model for memory pruning"""
    
    status: str = Field(..., description="Pruning status")
    items_pruned: int = Field(0, ge=0, description="Number of items pruned")
    items_remaining: int = Field(0, ge=0, description="Number of items remaining")
    space_freed: Optional[str] = Field(None, description="Estimated space freed")


# Configuration Models
class APIConfigInfo(BaseAPIModel):
    """API configuration information"""
    
    version: str = Field(..., description="API version")
    title: str = Field(..., description="API title")
    features: List[str] = Field(..., description="Enabled features")
    limits: Dict[str, Any] = Field(..., description="API limits and quotas")
    endpoints: List[str] = Field(..., description="Available endpoints")


# Pagination Models
class PaginationInfo(BaseAPIModel):
    """Pagination information"""
    
    page: int = Field(1, ge=1, description="Current page number")
    size: int = Field(20, ge=1, le=100, description="Page size")
    total: int = Field(0, ge=0, description="Total number of items")
    pages: int = Field(0, ge=0, description="Total number of pages")
    has_next: bool = Field(False, description="Whether there is a next page")
    has_prev: bool = Field(False, description="Whether there is a previous page")


class PaginatedResponse(BaseAPIModel):
    """Base paginated response model"""
    
    items: List[Any] = Field(..., description="Response items")
    pagination: PaginationInfo = Field(..., description="Pagination information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# Export all models
__all__ = [
    # Enums
    "SystemStatus",
    "ComponentStatus", 
    "QueryType",
    "MessageType",
    
    # Base Models
    "BaseAPIModel",
    "TimestampedModel",
    
    # Request Models
    "QueryRequest",
    "WebSocketMessage",
    "BrainRestartRequest",
    "MemoryPruningRequest",
    
    # Response Models
    "SourceInfo",
    "QueryResponse",
    "ComponentHealth",
    "SystemMetrics",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
    
    # Specialized Response Models
    "ProtocolInfoResponse",
    "PoolAnalysisResponse",
    "VotingInfoResponse",
    
    # WebSocket Models
    "WebSocketQueryResult",
    "WebSocketError",
    "WebSocketNotification",
    
    # Utility Models
    "QueryValidation",
    "RateLimitInfo",
    "APIConfigInfo",
    "PaginationInfo",
    "PaginatedResponse",
    
    # Admin Models
    "BrainRestartResponse",
    "MemoryPruningResponse"
]