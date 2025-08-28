"""
CDP Configuration

Configuration settings for Coinbase Developer Platform SDK integration
including network settings, wallet configuration, and operational parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os


@dataclass 
class CDPConfig:
    """Configuration for CDP SDK operations"""
    
    # CDP API Configuration
    api_key_name: str = ""
    api_key_private_key: str = ""
    network_id: str = "base-mainnet"  # Base network for Aerodrome
    
    # Wallet Configuration
    wallet_id: Optional[str] = None
    wallet_seed: Optional[str] = None
    
    # Transaction Settings
    max_gas_price: float = 50.0  # gwei
    gas_multiplier: float = 1.1
    confirmation_blocks: int = 1
    transaction_timeout: int = 300  # seconds
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    exponential_backoff: bool = True
    
    # Safety Settings
    simulation_required: bool = True
    max_slippage: float = 0.02  # 2%
    min_confirmations: int = 1
    
    # Rate Limiting
    requests_per_second: int = 10
    burst_limit: int = 50
    
    # Monitoring
    enable_metrics: bool = True
    log_transactions: bool = True
    
    @classmethod
    def from_env(cls) -> "CDPConfig":
        """Create configuration from environment variables"""
        return cls(
            api_key_name=os.getenv("CDP_API_KEY_NAME", ""),
            api_key_private_key=os.getenv("CDP_API_KEY_PRIVATE_KEY", ""),
            network_id=os.getenv("CDP_NETWORK_ID", "base-mainnet"),
            wallet_id=os.getenv("CDP_WALLET_ID"),
            wallet_seed=os.getenv("CDP_WALLET_SEED"),
            max_gas_price=float(os.getenv("CDP_MAX_GAS_PRICE", "50.0")),
            gas_multiplier=float(os.getenv("CDP_GAS_MULTIPLIER", "1.1")),
            confirmation_blocks=int(os.getenv("CDP_CONFIRMATION_BLOCKS", "1")),
            transaction_timeout=int(os.getenv("CDP_TRANSACTION_TIMEOUT", "300")),
            max_retries=int(os.getenv("CDP_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("CDP_RETRY_DELAY", "1.0")),
            simulation_required=os.getenv("CDP_SIMULATION_REQUIRED", "true").lower() == "true",
            max_slippage=float(os.getenv("CDP_MAX_SLIPPAGE", "0.02")),
            min_confirmations=int(os.getenv("CDP_MIN_CONFIRMATIONS", "1")),
            requests_per_second=int(os.getenv("CDP_REQUESTS_PER_SECOND", "10")),
            burst_limit=int(os.getenv("CDP_BURST_LIMIT", "50")),
            enable_metrics=os.getenv("CDP_ENABLE_METRICS", "true").lower() == "true",
            log_transactions=os.getenv("CDP_LOG_TRANSACTIONS", "true").lower() == "true"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if not self.api_key_name:
            raise ValueError("CDP API key name is required")
        
        if not self.api_key_private_key:
            raise ValueError("CDP API key private key is required")
        
        if self.max_gas_price <= 0:
            raise ValueError("max_gas_price must be positive")
        
        if self.gas_multiplier <= 0:
            raise ValueError("gas_multiplier must be positive")
        
        if self.confirmation_blocks < 0:
            raise ValueError("confirmation_blocks must be non-negative")
        
        if self.transaction_timeout <= 0:
            raise ValueError("transaction_timeout must be positive")
        
        if self.max_slippage < 0 or self.max_slippage > 1:
            raise ValueError("max_slippage must be between 0 and 1")
        
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        
        if self.burst_limit <= 0:
            raise ValueError("burst_limit must be positive")