"""
Settings - Environment-based configuration management

Centralized settings management using environment variables with sensible defaults.
"""

from typing import Optional, Dict, Any
import os
from pathlib import Path

from ..brain.config import BrainConfig
from ..memory.config import MemoryConfig
from ..cdp.config import CDPConfig


class Settings:
    """Environment-based configuration management"""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize settings from environment variables"""
        
        # Load .env file if specified
        if env_file:
            self.load_env_file(env_file)
        
        # Core settings
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.cdp_api_key_name = os.getenv("CDP_API_KEY_NAME", "")
        self.cdp_api_key_private_key = os.getenv("CDP_API_KEY_PRIVATE_KEY", "")
        
        # Network Configuration
        self.network = os.getenv("NETWORK", "base-mainnet")
        self.rpc_url = os.getenv("RPC_URL", "")
        
        # Agent Configuration
        self.agent_name = os.getenv("AGENT_NAME", "aerodrome-agent")
        self.operation_mode = os.getenv("OPERATION_MODE", "autonomous")  # autonomous, manual, simulation
        self.risk_level = os.getenv("RISK_LEVEL", "medium")  # low, medium, high
        
        # Performance
        self.max_concurrent_operations = int(os.getenv("MAX_CONCURRENT_OPERATIONS", "5"))
        self.operation_interval = int(os.getenv("OPERATION_INTERVAL", "60"))  # seconds
        
        # Safety
        self.emergency_stop_enabled = os.getenv("EMERGENCY_STOP_ENABLED", "true").lower() == "true"
        self.max_loss_percentage = float(os.getenv("MAX_LOSS_PERCENTAGE", "10.0"))
        
        # Monitoring
        self.metrics_enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"
        self.metrics_port = int(os.getenv("METRICS_PORT", "8080"))
        self.health_check_port = int(os.getenv("HEALTH_CHECK_PORT", "8081"))
        
        # Storage
        self.data_dir = Path(os.getenv("DATA_DIR", "./data"))
        self.logs_dir = Path(os.getenv("LOGS_DIR", "./logs"))
        self.cache_dir = Path(os.getenv("CACHE_DIR", "./cache"))
        
        # Ensure directories exist
        for directory in [self.data_dir, self.logs_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_env_file(self, env_file: str) -> None:
        """Load environment variables from file"""
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass  # dotenv not available
    
    def get_brain_config(self) -> BrainConfig:
        """Get brain configuration"""
        return BrainConfig(
            confidence_threshold=float(os.getenv("BRAIN_CONFIDENCE_THRESHOLD", "0.7")),
            risk_threshold=float(os.getenv("BRAIN_RISK_THRESHOLD", "0.3")),
            min_opportunity_score=float(os.getenv("BRAIN_MIN_OPPORTUNITY_SCORE", "0.6")),
            observation_interval=int(os.getenv("BRAIN_OBSERVATION_INTERVAL", "60")),
            execution_timeout=int(os.getenv("BRAIN_EXECUTION_TIMEOUT", "30")),
            max_position_size=float(os.getenv("BRAIN_MAX_POSITION_SIZE", "0.2")),
            max_slippage=float(os.getenv("BRAIN_MAX_SLIPPAGE", "0.02")),
            emergency_stop_loss=float(os.getenv("BRAIN_EMERGENCY_STOP_LOSS", "0.1")),
            max_gas_price=float(os.getenv("BRAIN_MAX_GAS_PRICE", "50.0")),
            log_level=self.log_level
        )
    
    def get_memory_config(self) -> MemoryConfig:
        """Get memory configuration"""
        return MemoryConfig.from_env()
    
    def get_cdp_config(self) -> CDPConfig:
        """Get CDP configuration"""
        return CDPConfig.from_env()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "agent_name": self.agent_name,
            "operation_mode": self.operation_mode,
            "risk_level": self.risk_level,
            "network": self.network,
            "max_concurrent_operations": self.max_concurrent_operations,
            "operation_interval": self.operation_interval,
            "emergency_stop_enabled": self.emergency_stop_enabled,
            "max_loss_percentage": self.max_loss_percentage,
            "metrics_enabled": self.metrics_enabled,
            "metrics_port": self.metrics_port,
            "health_check_port": self.health_check_port,
            "data_dir": str(self.data_dir),
            "logs_dir": str(self.logs_dir),
            "cache_dir": str(self.cache_dir)
        }
    
    def validate(self) -> None:
        """Validate configuration"""
        errors = []
        
        # Check required API keys
        if not self.openai_api_key and self.environment != "test":
            errors.append("OPENAI_API_KEY is required")
        
        if not self.cdp_api_key_name and self.environment != "test":
            errors.append("CDP_API_KEY_NAME is required")
        
        if not self.cdp_api_key_private_key and self.environment != "test":
            errors.append("CDP_API_KEY_PRIVATE_KEY is required")
        
        # Validate ranges
        if not 0 <= self.max_loss_percentage <= 100:
            errors.append("MAX_LOSS_PERCENTAGE must be between 0 and 100")
        
        if self.max_concurrent_operations <= 0:
            errors.append("MAX_CONCURRENT_OPERATIONS must be positive")
        
        if self.operation_interval <= 0:
            errors.append("OPERATION_INTERVAL must be positive")
        
        # Validate operation mode
        valid_modes = ["autonomous", "manual", "simulation"]
        if self.operation_mode not in valid_modes:
            errors.append(f"OPERATION_MODE must be one of {valid_modes}")
        
        # Validate risk level
        valid_risk_levels = ["low", "medium", "high"]
        if self.risk_level not in valid_risk_levels:
            errors.append(f"RISK_LEVEL must be one of {valid_risk_levels}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    @property
    def is_test(self) -> bool:
        """Check if running in test mode"""
        return self.environment == "test"