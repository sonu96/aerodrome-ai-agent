"""
Aerodrome AI Agent API Package

FastAPI-based REST API and WebSocket server for the Aerodrome AI Agent.

This package provides:
- REST endpoints for querying the brain system
- WebSocket support for real-time updates
- Health checks and metrics
- Authentication and rate limiting
- Production-ready configuration

Usage:
    from src.api.main import app
    
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""

__version__ = "1.0.0"
__author__ = "Aerodrome AI Team"

from .main import app
from .models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    MetricsResponse,
    SystemStatus,
    ComponentStatus
)

__all__ = [
    "app",
    "QueryRequest", 
    "QueryResponse",
    "HealthResponse",
    "MetricsResponse",
    "SystemStatus",
    "ComponentStatus"
]