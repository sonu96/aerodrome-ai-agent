"""
Aerodrome Protocol Integration Package

This package provides comprehensive integration with the Aerodrome DEX protocol
on Base mainnet using QuickNode's Aerodrome API addon.

Components:
- AerodromeClient: Core API client for protocol interactions
- PoolMonitor: Real-time pool monitoring and analytics  
- VotingAnalyzer: veAERO voting analysis and insights

Usage:
    from src.protocol import AerodromeClient, PoolMonitor, VotingAnalyzer
    
    # Initialize client
    async with AerodromeClient(quicknode_url) as client:
        # Use client for protocol operations
        pools = await client.search_pools(min_tvl=100000)
        
        # Set up monitoring
        monitor = PoolMonitor(client)
        await monitor.start_monitoring(pools[0].address)
        
        # Analyze voting
        analyzer = VotingAnalyzer(client)
        analytics = await analyzer.generate_voting_analytics(current_epoch)
"""

from .aerodrome_client import (
    AerodromeClient,
    SwapType,
    PoolType,
    TokenInfo,
    PoolInfo,
    SwapQuote,
    Transaction,
    AERODROME_CONTRACTS,
    BASE_MAINNET_CONFIG
)

from .pool_monitor import (
    PoolMonitor,
    PoolMetrics,
    PoolAlert,
    MonitoringConfig,
    AlertType,
    MonitoringStatus,
    PoolAnalyzer
)

from .voting_analyzer import (
    VotingAnalyzer,
    VoterInfo,
    GaugeInfo,
    BribeInfo,
    VotingRound,
    VotingAnalytics,
    VoteType,
    BribeType,
    VoterCategory,
    VotingPatternAnalyzer
)

__version__ = "1.0.0"

__all__ = [
    # Core client
    "AerodromeClient",
    "SwapType", 
    "PoolType",
    "TokenInfo",
    "PoolInfo", 
    "SwapQuote",
    "Transaction",
    "AERODROME_CONTRACTS",
    "BASE_MAINNET_CONFIG",
    
    # Pool monitoring
    "PoolMonitor",
    "PoolMetrics",
    "PoolAlert", 
    "MonitoringConfig",
    "AlertType",
    "MonitoringStatus",
    "PoolAnalyzer",
    
    # Voting analysis
    "VotingAnalyzer",
    "VoterInfo",
    "GaugeInfo", 
    "BribeInfo",
    "VotingRound",
    "VotingAnalytics",
    "VoteType",
    "BribeType", 
    "VoterCategory",
    "VotingPatternAnalyzer"
]