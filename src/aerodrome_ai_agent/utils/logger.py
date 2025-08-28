"""
Centralized Logging Configuration for Aerodrome AI Agent

This module provides centralized logging setup with:
- Structured logging with JSON formatting for production
- Console logging for development
- File rotation for persistent logs
- Performance and security event logging
- Different log levels for different components
"""

import logging
import logging.config
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import os

from ..config.settings import Settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname",
                          "filename", "module", "lineno", "funcName", "created", "msecs",
                          "relativeCreated", "thread", "threadName", "processName",
                          "process", "exc_info", "exc_text", "stack_info"]:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class AgentLogger:
    """Centralized logger management for Aerodrome AI Agent"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.log_dir = Path(self.settings.log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self._setup_logging()
        
        # Get main logger
        self.logger = logging.getLogger("aerodrome_ai_agent")
        self.logger.info("Centralized logging system initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        
        # Determine log level
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        
        # Create logging configuration
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "()": StructuredFormatter
                },
                "console": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "file": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "console" if self.settings.environment == "development" else "structured",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": log_level,
                    "formatter": "structured" if self.settings.environment == "production" else "file",
                    "filename": str(self.log_dir / "agent.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": logging.ERROR,
                    "formatter": "structured",
                    "filename": str(self.log_dir / "agent_errors.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 3
                },
                "security": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": logging.WARNING,
                    "formatter": "structured",
                    "filename": str(self.log_dir / "security.log"),
                    "maxBytes": 5242880,  # 5MB
                    "backupCount": 10
                },
                "performance": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": logging.INFO,
                    "formatter": "structured", 
                    "filename": str(self.log_dir / "performance.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 3
                }
            },
            "loggers": {
                "aerodrome_ai_agent": {
                    "level": log_level,
                    "handlers": ["console", "file", "error_file"],
                    "propagate": False
                },
                "aerodrome_ai_agent.brain": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "aerodrome_ai_agent.memory": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "aerodrome_ai_agent.cdp": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "aerodrome_ai_agent.orchestrator": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "aerodrome_ai_agent.monitoring": {
                    "level": log_level,
                    "handlers": ["console", "file", "performance"],
                    "propagate": False
                },
                "aerodrome_ai_agent.security": {
                    "level": logging.WARNING,
                    "handlers": ["security", "error_file"],
                    "propagate": False
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["console"]
            }
        }
        
        # Apply configuration
        logging.config.dictConfig(config)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific component"""
        return logging.getLogger(f"aerodrome_ai_agent.{name}")
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        perf_logger = logging.getLogger("aerodrome_ai_agent.monitoring")
        perf_logger.info(
            f"Performance: {operation}",
            extra={
                "operation": operation,
                "duration_ms": duration * 1000,
                "performance_metric": True,
                **kwargs
            }
        )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events"""
        security_logger = logging.getLogger("aerodrome_ai_agent.security")
        security_logger.warning(
            f"Security Event: {event_type}",
            extra={
                "event_type": event_type,
                "security_event": True,
                "details": details
            }
        )
    
    def log_trade_execution(self, trade_data: Dict[str, Any]):
        """Log trade execution details"""
        logger = logging.getLogger("aerodrome_ai_agent.brain")
        logger.info(
            f"Trade executed: {trade_data.get('action_type', 'unknown')}",
            extra={
                "trade_data": trade_data,
                "trade_execution": True
            }
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context"""
        logger = logging.getLogger("aerodrome_ai_agent")
        logger.error(
            f"Error occurred: {str(error)}",
            exc_info=error,
            extra={
                "error_context": context,
                "error_type": type(error).__name__
            }
        )
    
    def log_system_metric(self, metric_name: str, value: float, **metadata):
        """Log system performance metrics"""
        perf_logger = logging.getLogger("aerodrome_ai_agent.monitoring")
        perf_logger.info(
            f"System Metric: {metric_name} = {value}",
            extra={
                "metric_name": metric_name,
                "metric_value": value,
                "system_metric": True,
                **metadata
            }
        )
    
    @staticmethod
    def setup_basic_logging(level: str = "INFO"):
        """Setup basic logging for simple use cases"""
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )


# Global logger instance
_logger_instance: Optional[AgentLogger] = None


def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance for a component"""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = AgentLogger()
    
    if name:
        return _logger_instance.get_logger(name)
    else:
        return _logger_instance.logger


def initialize_logging(settings: Settings) -> AgentLogger:
    """Initialize the centralized logging system"""
    global _logger_instance
    _logger_instance = AgentLogger(settings)
    return _logger_instance


def log_performance(operation: str, duration: float, **kwargs):
    """Helper function to log performance metrics"""
    global _logger_instance
    if _logger_instance:
        _logger_instance.log_performance(operation, duration, **kwargs)


def log_security_event(event_type: str, details: Dict[str, Any]):
    """Helper function to log security events"""
    global _logger_instance
    if _logger_instance:
        _logger_instance.log_security_event(event_type, details)


def log_trade_execution(trade_data: Dict[str, Any]):
    """Helper function to log trade executions"""
    global _logger_instance
    if _logger_instance:
        _logger_instance.log_trade_execution(trade_data)


def log_error_with_context(error: Exception, context: Dict[str, Any]):
    """Helper function to log errors with context"""
    global _logger_instance
    if _logger_instance:
        _logger_instance.log_error_with_context(error, context)


def log_system_metric(metric_name: str, value: float, **metadata):
    """Helper function to log system metrics"""
    global _logger_instance
    if _logger_instance:
        _logger_instance.log_system_metric(metric_name, value, **metadata)