"""
Brain Configuration

Defines configuration settings for the Aerodrome brain operations including
thresholds, timeouts, safety limits, and operational parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BrainConfig:
    """Configuration for the Aerodrome Brain"""
    
    # Decision thresholds
    confidence_threshold: float = 0.7
    risk_threshold: float = 0.3  
    min_opportunity_score: float = 0.6
    
    # Timing parameters
    observation_interval: int = 60  # seconds
    execution_timeout: int = 30  # seconds
    decision_timeout: int = 10  # seconds
    
    # Memory settings
    max_memories_per_query: int = 10
    pattern_extraction_threshold: int = 5
    memory_recall_timeout: int = 5  # seconds
    
    # Safety limits
    max_position_size: float = 0.2  # 20% of portfolio
    max_slippage: float = 0.02  # 2%
    emergency_stop_loss: float = 0.1  # 10%
    max_gas_price: float = 50.0  # gwei
    
    # Portfolio management
    min_trade_amount: float = 10.0  # USD
    max_trade_amount: float = 10000.0  # USD
    rebalance_threshold: float = 0.05  # 5%
    
    # Risk management
    max_drawdown: float = 0.15  # 15%
    volatility_threshold: float = 0.3  # 30%
    correlation_limit: float = 0.8
    
    # Operational settings
    max_concurrent_positions: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Logging and monitoring
    log_level: str = "INFO"
    metrics_interval: int = 300  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BrainConfig":
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if self.risk_threshold < 0 or self.risk_threshold > 1:
            raise ValueError("risk_threshold must be between 0 and 1")
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("max_position_size must be between 0 and 1")
        
        if self.max_slippage < 0 or self.max_slippage > 1:
            raise ValueError("max_slippage must be between 0 and 1")
        
        if self.observation_interval <= 0:
            raise ValueError("observation_interval must be positive")
        
        if self.execution_timeout <= 0:
            raise ValueError("execution_timeout must be positive")