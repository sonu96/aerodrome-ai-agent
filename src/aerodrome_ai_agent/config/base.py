"""
Base Configuration - Main agent configuration class

Centralized configuration management combining all component configurations.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from .settings import Settings
from ..brain.config import BrainConfig
from ..memory.config import MemoryConfig
from ..cdp.config import CDPConfig


@dataclass
class AgentConfig:
    """Main agent configuration combining all components"""
    
    # Core settings
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    brain_config: Optional[BrainConfig] = None
    memory_config: Optional[MemoryConfig] = None
    cdp_config: Optional[CDPConfig] = None
    
    # Agent-specific settings
    agent_name: str = "aerodrome-agent"
    operation_mode: str = "simulation"  # autonomous, simulation, manual
    risk_level: str = "medium"  # low, medium, high
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "AgentConfig":
        """Create configuration from environment"""
        
        settings = Settings(env_file)
        
        return cls(
            environment=settings.environment,
            debug=settings.debug,
            log_level=settings.log_level,
            brain_config=settings.get_brain_config(),
            memory_config=settings.get_memory_config(),
            cdp_config=settings.get_cdp_config(),
            agent_name=settings.agent_name,
            operation_mode=settings.operation_mode,
            risk_level=settings.risk_level
        )
    
    def get_brain_config(self) -> BrainConfig:
        """Get brain configuration"""
        return self.brain_config or BrainConfig()
    
    def get_memory_config(self) -> MemoryConfig:
        """Get memory configuration"""
        return self.memory_config or MemoryConfig()
    
    def get_cdp_config(self) -> CDPConfig:
        """Get CDP configuration"""
        return self.cdp_config or CDPConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "agent_name": self.agent_name,
            "operation_mode": self.operation_mode,
            "risk_level": self.risk_level
        }
    
    def validate(self) -> None:
        """Validate configuration"""
        if self.brain_config:
            self.brain_config.validate()
        if self.memory_config:
            self.memory_config.validate()
        if self.cdp_config:
            self.cdp_config.validate()