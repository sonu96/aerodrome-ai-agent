"""
Configuration Module - Centralized configuration management

This module provides centralized configuration management for the Aerodrome AI Agent,
including environment-based configuration loading and validation.

Key Components:
- AgentConfig: Main agent configuration
- Environment-based configuration loading
- Configuration validation and defaults
"""

from .base import AgentConfig
from .settings import Settings

__all__ = [
    "AgentConfig",
    "Settings",
]