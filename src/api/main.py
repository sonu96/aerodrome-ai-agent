"""
Aerodrome AI Agent FastAPI Application

Production-ready API server with:
- REST endpoints for querying the brain
- WebSocket support for real-time updates
- Health check and metrics endpoints
- Authentication middleware
- CORS configuration
- Rate limiting
- Prometheus metrics
- Graceful shutdown
"""

import asyncio
import json
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import structlog
import uvicorn
from fastapi import (
    FastAPI, 
    HTTPException, 
    Depends, 
    WebSocket, 
    WebSocketDisconnect,
    Request,
    Response,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter, 
    Histogram, 
    Gauge, 
    generate_latest, 
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    MetricsResponse,
    WebSocketMessage,
    ErrorResponse,
    SystemStatus
)
from ..brain.core import AerodromeBrain, BrainConfig, SystemStatus as BrainSystemStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# Prometheus metrics
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('api_request_duration_seconds', 'API request duration')
active_connections = Gauge('api_websocket_connections', 'Active WebSocket connections')
query_count = Counter('brain_queries_total', 'Total brain queries', ['status'])
query_duration = Histogram('brain_query_duration_seconds', 'Brain query processing time')
system_health = Gauge('system_health_status', 'System health status (0=error, 1=degraded, 2=healthy)')

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# Global brain instance
brain: Optional[AerodromeBrain] = None

# WebSocket connections
websocket_connections: Set[WebSocket] = set()

class Config:
    """Application configuration"""
    
    # API Configuration
    API_TITLE = os.getenv("API_TITLE", "Aerodrome AI Agent API")
    API_VERSION = os.getenv("API_VERSION", "1.0.0")
    API_DESCRIPTION = os.getenv("API_DESCRIPTION", "AI-powered Aerodrome Protocol assistant")
    
    # Security
    API_KEY = os.getenv("API_KEY", "")
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    TRUSTED_HOSTS = os.getenv("TRUSTED_HOSTS", "*").split(",")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = os.getenv("RATE_LIMIT_REQUESTS", "100/minute")
    RATE_LIMIT_QUERY = os.getenv("RATE_LIMIT_QUERY", "10/minute")
    
    # Brain Configuration
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
    MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "10"))
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    
    # Memory Configuration
    MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
    NEO4J_URI = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    
    # Protocol Configuration
    QUICKNODE_URL = os.getenv("QUICKNODE_URL", "")
    
    # Intelligence Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key authentication"""
    if not Config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured"
        )
    
    if credentials.credentials != Config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return credentials.credentials


async def get_brain() -> AerodromeBrain:
    """Get the global brain instance"""
    global brain
    if brain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Brain not initialized"
        )
    return brain


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global brain
    
    try:
        # Initialize brain
        logger.info("Initializing Aerodrome Brain...")
        
        brain_config = BrainConfig(
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            max_concurrent_queries=Config.MAX_CONCURRENT_QUERIES,
            health_check_interval=Config.HEALTH_CHECK_INTERVAL,
            memory_config={
                "api_key": Config.MEM0_API_KEY,
                "enable_graph": True,
                "neo4j_config": {
                    "uri": Config.NEO4J_URI,
                    "username": Config.NEO4J_USERNAME,
                    "password": Config.NEO4J_PASSWORD
                }
            },
            protocol_config={
                "quicknode_url": Config.QUICKNODE_URL
            },
            intelligence_config={
                "gemini_api_key": Config.GEMINI_API_KEY,
                "model": Config.GEMINI_MODEL
            }
        )
        
        brain = AerodromeBrain(brain_config)
        await brain.initialize()
        
        logger.info("Brain initialized successfully")
        
        # Setup graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            asyncio.create_task(shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        yield
        
    finally:
        # Cleanup
        await shutdown()


async def shutdown():
    """Graceful shutdown"""
    global brain, websocket_connections
    
    logger.info("Starting graceful shutdown...")
    
    # Close WebSocket connections
    for ws in websocket_connections.copy():
        try:
            await ws.close(code=1000, reason="Server shutting down")
        except Exception as e:
            logger.warning(f"Error closing WebSocket: {e}")
    
    # Shutdown brain
    if brain:
        await brain.shutdown()
        brain = None
    
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description=Config.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if "*" not in Config.TRUSTED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=Config.TRUSTED_HOSTS
    )

app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect metrics for each request"""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    request_duration.observe(duration)
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests and responses"""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None
    )
    
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration=duration
    )
    
    return response


# Health Check Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    try:
        brain_instance = await get_brain()
        brain_health = await brain_instance.get_system_health()
        
        # Map brain status to API status
        brain_status = brain_health["system_status"]
        if brain_status == BrainSystemStatus.HEALTHY.value:
            status_enum = SystemStatus.HEALTHY
        elif brain_status == BrainSystemStatus.DEGRADED.value:
            status_enum = SystemStatus.DEGRADED
        else:
            status_enum = SystemStatus.ERROR
        
        # Update Prometheus health metric
        health_value = {"healthy": 2, "degraded": 1, "error": 0}
        system_health.set(health_value.get(status_enum.value, 0))
        
        return HealthResponse(
            status=status_enum,
            timestamp=datetime.now(),
            version=Config.API_VERSION,
            uptime=brain_health["uptime"],
            components=brain_health["component_health"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        system_health.set(0)  # Error state
        return HealthResponse(
            status=SystemStatus.ERROR,
            timestamp=datetime.now(),
            version=Config.API_VERSION,
            error=str(e)
        )


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Kubernetes readiness probe"""
    try:
        brain_instance = await get_brain()
        health = await brain_instance.get_system_health()
        
        if health["system_status"] in [BrainSystemStatus.HEALTHY.value, BrainSystemStatus.DEGRADED.value]:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.now()}


# Metrics Endpoint
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/detailed", response_model=MetricsResponse, tags=["Monitoring"])
@limiter.limit(Config.RATE_LIMIT_REQUESTS)
async def get_detailed_metrics(request: Request, _: str = Depends(verify_api_key)):
    """Get detailed system metrics"""
    try:
        brain_instance = await get_brain()
        health = await brain_instance.get_system_health()
        
        return MetricsResponse(
            timestamp=datetime.now(),
            system_metrics=health["metrics"],
            component_health=health["component_health"],
            api_metrics={
                "active_websocket_connections": len(websocket_connections)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get detailed metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


# Query Endpoints
@app.post("/query", response_model=QueryResponse, tags=["Query"])
@limiter.limit(Config.RATE_LIMIT_QUERY)
async def process_query(
    request: Request,
    query_request: QueryRequest,
    _: str = Depends(verify_api_key)
):
    """Process a natural language query"""
    start_time = time.time()
    
    try:
        brain_instance = await get_brain()
        
        # Process query through brain
        response = await brain_instance.process_query(
            query=query_request.query,
            context=query_request.context,
            user_id=query_request.user_id
        )
        
        # Record metrics
        duration = time.time() - start_time
        query_duration.observe(duration)
        query_count.labels(status="success").inc()
        
        # Convert brain response to API response
        api_response = QueryResponse(
            query_id=response.query_id,
            query=query_request.query,
            response=response.response,
            confidence=response.confidence,
            sources=response.sources,
            metadata=response.metadata,
            timestamp=datetime.now(),
            processing_time=duration
        )
        
        # Broadcast to WebSocket clients if confidence is high
        if response.confidence >= Config.CONFIDENCE_THRESHOLD:
            await broadcast_to_websockets({
                "type": "query_result",
                "data": api_response.dict()
            })
        
        return api_response
        
    except Exception as e:
        duration = time.time() - start_time
        query_duration.observe(duration)
        query_count.labels(status="error").inc()
        
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/query/{query_id}", response_model=QueryResponse, tags=["Query"])
@limiter.limit(Config.RATE_LIMIT_REQUESTS)
async def get_query_result(
    request: Request,
    query_id: str,
    _: str = Depends(verify_api_key)
):
    """Get cached query result by ID"""
    # This would require implementing query result caching in the brain
    raise HTTPException(status_code=501, detail="Query result caching not implemented")


# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.add(websocket)
    active_connections.set(len(websocket_connections))
    
    logger.info("WebSocket client connected")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to Aerodrome AI Agent",
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Receive messages from client
            try:
                data = await websocket.receive_json()
                message = WebSocketMessage(**data)
                
                # Handle different message types
                if message.type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif message.type == "query":
                    # Process query and send result
                    try:
                        brain_instance = await get_brain()
                        response = await brain_instance.process_query(
                            query=message.data.get("query", ""),
                            context=message.data.get("context"),
                            user_id=message.data.get("user_id")
                        )
                        
                        await websocket.send_json({
                            "type": "query_result",
                            "data": {
                                "query_id": response.query_id,
                                "response": response.response,
                                "confidence": response.confidence,
                                "sources": response.sources,
                                "metadata": response.metadata
                            },
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": str(e)},
                            "timestamp": datetime.now().isoformat()
                        })
                
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_connections.discard(websocket)
        active_connections.set(len(websocket_connections))


async def broadcast_to_websockets(message: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients"""
    if not websocket_connections:
        return
    
    disconnected = set()
    
    for websocket in websocket_connections:
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
            disconnected.add(websocket)
    
    # Remove disconnected clients
    for ws in disconnected:
        websocket_connections.discard(ws)
    
    active_connections.set(len(websocket_connections))


# Admin Endpoints
@app.post("/admin/brain/restart", tags=["Admin"])
@limiter.limit("5/minute")
async def restart_brain(request: Request, _: str = Depends(verify_api_key)):
    """Restart the brain system"""
    try:
        global brain
        
        if brain:
            await brain.shutdown()
        
        # Reinitialize brain
        brain_config = BrainConfig(
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            max_concurrent_queries=Config.MAX_CONCURRENT_QUERIES,
            health_check_interval=Config.HEALTH_CHECK_INTERVAL,
            memory_config={
                "api_key": Config.MEM0_API_KEY,
                "enable_graph": True,
                "neo4j_config": {
                    "uri": Config.NEO4J_URI,
                    "username": Config.NEO4J_USERNAME,
                    "password": Config.NEO4J_PASSWORD
                }
            },
            protocol_config={
                "quicknode_url": Config.QUICKNODE_URL
            },
            intelligence_config={
                "gemini_api_key": Config.GEMINI_API_KEY,
                "model": Config.GEMINI_MODEL
            }
        )
        
        brain = AerodromeBrain(brain_config)
        await brain.initialize()
        
        return {"status": "success", "message": "Brain restarted successfully"}
        
    except Exception as e:
        logger.error(f"Brain restart failed: {e}")
        raise HTTPException(status_code=500, detail=f"Brain restart failed: {str(e)}")


@app.post("/admin/memory/prune", tags=["Admin"])
@limiter.limit("1/hour")
async def trigger_memory_pruning(request: Request, _: str = Depends(verify_api_key)):
    """Manually trigger memory pruning"""
    try:
        brain_instance = await get_brain()
        
        if brain_instance.pruning_engine:
            await brain_instance.pruning_engine.run_pruning_cycle()
            return {"status": "success", "message": "Memory pruning completed"}
        else:
            raise HTTPException(status_code=503, detail="Memory pruning engine not available")
            
    except Exception as e:
        logger.error(f"Memory pruning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory pruning failed: {str(e)}")


# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            status_code=500,
            timestamp=datetime.now()
        ).dict()
    )


# Application entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("ENV", "production") != "production",
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False
    )