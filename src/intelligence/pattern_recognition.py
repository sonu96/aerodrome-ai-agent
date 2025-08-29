"""
Pattern Recognition Engine

Advanced AI-powered pattern detection system for:
- Trading patterns in pool data
- Voting coalitions detection
- Arbitrage opportunities recognition  
- Market correlations analysis
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import structlog
from .gemini_client import GeminiClient, GeminiModel, ModelConfig, FunctionSpec, StructuredOutput

logger = structlog.get_logger(__name__)


class PatternType(Enum):
    """Types of patterns that can be detected"""
    TRADING_PATTERN = "trading_pattern"
    VOTING_COALITION = "voting_coalition"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    MARKET_CORRELATION = "market_correlation"
    ANOMALY = "anomaly"
    SEASONAL = "seasonal"


class PatternConfidence(Enum):
    """Pattern confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PoolData:
    """Pool data structure for analysis"""
    pool_address: str
    token0_symbol: str
    token1_symbol: str
    reserves: Tuple[float, float]
    price: float
    volume_24h: float
    fees_24h: float
    liquidity: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_address": self.pool_address,
            "token0_symbol": self.token0_symbol,
            "token1_symbol": self.token1_symbol,
            "reserves": list(self.reserves),
            "price": self.price,
            "volume_24h": self.volume_24h,
            "fees_24h": self.fees_24h,
            "liquidity": self.liquidity,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class VotingData:
    """Voting data structure for coalition analysis"""
    proposal_id: str
    voter_address: str
    voting_power: float
    vote_choice: str  # "for", "against", "abstain"
    timestamp: datetime
    block_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "voter_address": self.voter_address,
            "voting_power": self.voting_power,
            "vote_choice": self.vote_choice,
            "timestamp": self.timestamp.isoformat(),
            "block_number": self.block_number
        }


class TradingPattern(StructuredOutput):
    """Detected trading pattern"""
    pattern_type: str = Field(..., description="Type of trading pattern")
    pattern_name: str = Field(..., description="Human-readable pattern name")
    description: str = Field(..., description="Detailed pattern description")
    pools_involved: List[str] = Field(..., description="Pool addresses involved")
    tokens_involved: List[str] = Field(..., description="Token symbols involved")
    timeframe: str = Field(..., description="Pattern timeframe (e.g., '1h', '4h', '1d')")
    strength: float = Field(..., ge=0.0, le=1.0, description="Pattern strength score")
    volume_impact: float = Field(..., description="Volume impact in USD")
    price_impact: float = Field(..., description="Price impact percentage")
    duration: str = Field(..., description="Expected pattern duration")
    next_action: str = Field(..., description="Suggested next action")
    risk_level: str = Field(..., description="Risk assessment")


class VotingCoalition(StructuredOutput):
    """Detected voting coalition"""
    coalition_id: str = Field(..., description="Unique coalition identifier")
    coalition_name: str = Field(..., description="Descriptive coalition name")
    members: List[str] = Field(..., description="Voter addresses in coalition")
    total_voting_power: float = Field(..., description="Combined voting power")
    voting_pattern: str = Field(..., description="Consistent voting pattern")
    proposals_aligned: List[str] = Field(..., description="Proposals where coalition voted together")
    alignment_percentage: float = Field(..., ge=0.0, le=100.0, description="Voting alignment percentage")
    influence_score: float = Field(..., ge=0.0, le=1.0, description="Coalition influence score")
    formation_date: str = Field(..., description="Estimated coalition formation date")
    activity_level: str = Field(..., description="Coalition activity level")
    motivations: List[str] = Field(..., description="Inferred motivations")


class ArbitrageOpportunity(StructuredOutput):
    """Detected arbitrage opportunity"""
    opportunity_id: str = Field(..., description="Unique opportunity identifier")
    opportunity_type: str = Field(..., description="Type of arbitrage")
    pools_involved: List[str] = Field(..., description="Pool addresses involved")
    tokens_involved: List[str] = Field(..., description="Token symbols")
    price_difference: float = Field(..., description="Price difference percentage")
    potential_profit: float = Field(..., description="Potential profit in USD")
    required_capital: float = Field(..., description="Required capital in USD")
    roi_percentage: float = Field(..., description="Return on investment percentage")
    execution_path: List[str] = Field(..., description="Step-by-step execution path")
    gas_cost_estimate: float = Field(..., description="Estimated gas costs")
    time_sensitivity: str = Field(..., description="Time sensitivity level")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    slippage_tolerance: float = Field(..., description="Recommended slippage tolerance")


class MarketCorrelation(StructuredOutput):
    """Detected market correlation"""
    correlation_id: str = Field(..., description="Unique correlation identifier")
    correlation_type: str = Field(..., description="Type of correlation")
    assets_involved: List[str] = Field(..., description="Assets showing correlation")
    correlation_strength: float = Field(..., ge=-1.0, le=1.0, description="Correlation coefficient")
    correlation_duration: str = Field(..., description="Duration of correlation")
    statistical_significance: float = Field(..., description="P-value of correlation")
    market_conditions: str = Field(..., description="Market conditions during correlation")
    volume_correlation: float = Field(..., description="Volume correlation coefficient")
    leading_indicator: Optional[str] = Field(None, description="Asset that leads the correlation")
    lag_time: Optional[str] = Field(None, description="Time lag in correlation")
    breakout_signals: List[str] = Field(..., description="Signals that might break correlation")


class PatternRecognitionEngine:
    """
    Advanced pattern recognition engine using Gemini AI
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        analysis_window_hours: int = 24,
        min_confidence: float = 0.7,
        enable_advanced_analytics: bool = True
    ):
        """
        Initialize pattern recognition engine
        
        Args:
            gemini_client: Configured Gemini client
            analysis_window_hours: Analysis time window in hours
            min_confidence: Minimum confidence threshold
            enable_advanced_analytics: Enable advanced analytics features
        """
        self.gemini_client = gemini_client
        self.analysis_window_hours = analysis_window_hours
        self.min_confidence = min_confidence
        self.enable_advanced_analytics = enable_advanced_analytics
        
        # Pattern history for trend analysis
        self.pattern_history: Dict[PatternType, List[Any]] = {
            pattern_type: [] for pattern_type in PatternType
        }
        
        # Register analysis functions with Gemini
        self._register_analysis_functions()
        
        logger.info(
            "Pattern recognition engine initialized",
            analysis_window_hours=analysis_window_hours,
            min_confidence=min_confidence,
            advanced_analytics=enable_advanced_analytics
        )
    
    def _register_analysis_functions(self) -> None:
        """Register analysis functions for function calling"""
        
        # Statistical analysis function
        stats_function = FunctionSpec(
            name="calculate_statistics",
            description="Calculate statistical metrics for pattern analysis",
            parameters={
                "type": "object",
                "properties": {
                    "data_points": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of numerical data points"
                    },
                    "metric_type": {
                        "type": "string",
                        "enum": ["correlation", "volatility", "trend", "momentum"],
                        "description": "Type of statistical metric to calculate"
                    }
                },
                "required": ["data_points", "metric_type"]
            }
        )
        
        # Pattern validation function
        validation_function = FunctionSpec(
            name="validate_pattern",
            description="Validate identified pattern against historical data",
            parameters={
                "type": "object",
                "properties": {
                    "pattern_type": {
                        "type": "string",
                        "description": "Type of pattern to validate"
                    },
                    "confidence_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Initial confidence score"
                    },
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of supporting evidence"
                    }
                },
                "required": ["pattern_type", "confidence_score", "supporting_evidence"]
            }
        )
        
        # Market context function
        context_function = FunctionSpec(
            name="analyze_market_context",
            description="Analyze current market context for pattern interpretation",
            parameters={
                "type": "object",
                "properties": {
                    "timeframe": {
                        "type": "string",
                        "description": "Analysis timeframe"
                    },
                    "market_conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Current market conditions"
                    }
                },
                "required": ["timeframe", "market_conditions"]
            }
        )
        
        self.gemini_client.register_multiple_functions([
            stats_function,
            validation_function, 
            context_function
        ])
    
    async def detect_trading_patterns(
        self,
        pool_data_history: List[PoolData],
        pattern_types: Optional[List[str]] = None,
        min_volume_threshold: float = 10000.0
    ) -> List[TradingPattern]:
        """
        Detect trading patterns in pool data
        
        Args:
            pool_data_history: Historical pool data
            pattern_types: Specific pattern types to look for
            min_volume_threshold: Minimum volume threshold in USD
            
        Returns:
            List of detected trading patterns
        """
        logger.info(
            "Starting trading pattern detection",
            data_points=len(pool_data_history),
            min_volume=min_volume_threshold
        )
        
        # Filter data by volume threshold
        filtered_data = [
            pool for pool in pool_data_history
            if pool.volume_24h >= min_volume_threshold
        ]
        
        if not filtered_data:
            logger.warning("No pool data meets volume threshold")
            return []
        
        # Prepare data for analysis
        data_summary = self._prepare_pool_data_summary(filtered_data)
        
        # Create analysis prompt
        prompt = f"""
        Analyze the following DeFi pool data to detect trading patterns:
        
        Data Summary:
        {json.dumps(data_summary, indent=2)}
        
        Analysis Requirements:
        - Look for significant volume spikes, price movements, and liquidity changes
        - Identify potential front-running, sandwich attacks, or MEV activities
        - Detect cyclical patterns, trend reversals, or breakout patterns
        - Assess arbitrage activities between related pools
        - Consider unusual trading behaviors or market manipulations
        
        Focus on patterns with high confidence and significant market impact.
        Provide specific pool addresses and quantitative metrics where possible.
        
        {"Pattern types to focus on: " + str(pattern_types) if pattern_types else ""}
        """
        
        try:
            # Generate structured analysis
            patterns = await self.gemini_client.generate_structured_content(
                prompt=prompt,
                output_schema=List[TradingPattern],
                config=ModelConfig(temperature=0.3, max_output_tokens=4096),
                cache_key="trading_pattern_analysis"
            )
            
            # Filter by confidence
            high_confidence_patterns = [
                pattern for pattern in patterns
                if pattern.confidence >= self.min_confidence
            ]
            
            # Store patterns in history
            self.pattern_history[PatternType.TRADING_PATTERN].extend(high_confidence_patterns)
            
            logger.info(
                "Trading pattern detection completed",
                total_patterns=len(patterns),
                high_confidence_patterns=len(high_confidence_patterns)
            )
            
            return high_confidence_patterns
            
        except Exception as e:
            logger.error("Trading pattern detection failed", error=str(e))
            raise
    
    async def detect_voting_coalitions(
        self,
        voting_history: List[VotingData],
        min_proposals: int = 3,
        min_alignment: float = 0.8
    ) -> List[VotingCoalition]:
        """
        Detect voting coalitions in governance data
        
        Args:
            voting_history: Historical voting data
            min_proposals: Minimum proposals for coalition detection
            min_alignment: Minimum alignment percentage
            
        Returns:
            List of detected voting coalitions
        """
        logger.info(
            "Starting voting coalition detection",
            voting_records=len(voting_history),
            min_proposals=min_proposals,
            min_alignment=min_alignment
        )
        
        if len(voting_history) < min_proposals:
            logger.warning("Insufficient voting data for coalition detection")
            return []
        
        # Prepare voting data for analysis
        voting_summary = self._prepare_voting_data_summary(voting_history)
        
        # Create analysis prompt
        prompt = f"""
        Analyze the following governance voting data to detect voting coalitions:
        
        Voting Data Summary:
        {json.dumps(voting_summary, indent=2)}
        
        Coalition Detection Criteria:
        - Minimum {min_proposals} proposals with aligned voting
        - Minimum {min_alignment * 100}% alignment rate
        - Consider voting power concentration
        - Look for consistent voting patterns across time
        - Identify potential coordination or shared interests
        
        Analysis Focus:
        - Group voters by consistent alignment patterns
        - Calculate combined voting power of potential coalitions
        - Assess the impact of coalitions on proposal outcomes
        - Identify temporal patterns in coalition formation
        - Consider cross-proposal consistency and motivation patterns
        
        Provide quantitative metrics and specific voter addresses.
        """
        
        try:
            # Generate structured analysis
            coalitions = await self.gemini_client.generate_structured_content(
                prompt=prompt,
                output_schema=List[VotingCoalition],
                config=ModelConfig(temperature=0.2, max_output_tokens=4096),
                cache_key="voting_coalition_analysis"
            )
            
            # Filter by confidence and minimum alignment
            valid_coalitions = [
                coalition for coalition in coalitions
                if (coalition.confidence >= self.min_confidence and
                    coalition.alignment_percentage >= min_alignment * 100)
            ]
            
            # Store coalitions in history
            self.pattern_history[PatternType.VOTING_COALITION].extend(valid_coalitions)
            
            logger.info(
                "Voting coalition detection completed",
                total_coalitions=len(coalitions),
                valid_coalitions=len(valid_coalitions)
            )
            
            return valid_coalitions
            
        except Exception as e:
            logger.error("Voting coalition detection failed", error=str(e))
            raise
    
    async def detect_arbitrage_opportunities(
        self,
        pool_data: List[PoolData],
        min_profit_threshold: float = 50.0,
        max_gas_cost: float = 100.0
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities across pools
        
        Args:
            pool_data: Current pool data
            min_profit_threshold: Minimum profit threshold in USD
            max_gas_cost: Maximum acceptable gas cost in USD
            
        Returns:
            List of detected arbitrage opportunities
        """
        logger.info(
            "Starting arbitrage opportunity detection",
            pools_analyzed=len(pool_data),
            min_profit=min_profit_threshold,
            max_gas=max_gas_cost
        )
        
        if len(pool_data) < 2:
            logger.warning("Insufficient pool data for arbitrage detection")
            return []
        
        # Group pools by token pairs for comparison
        token_pair_pools = self._group_pools_by_token_pairs(pool_data)
        
        # Prepare arbitrage analysis data
        arbitrage_data = self._prepare_arbitrage_analysis_data(token_pair_pools)
        
        # Create analysis prompt
        prompt = f"""
        Analyze the following DeFi pool data to detect arbitrage opportunities:
        
        Pool Price Comparison Data:
        {json.dumps(arbitrage_data, indent=2)}
        
        Arbitrage Detection Criteria:
        - Minimum profit threshold: ${min_profit_threshold}
        - Maximum gas cost: ${max_gas_cost}
        - Consider liquidity depth and slippage
        - Account for transaction fees and MEV competition
        - Evaluate multi-hop arbitrage paths
        
        Analysis Focus:
        - Identify price discrepancies between pools with same token pairs
        - Calculate potential profit after fees and gas costs
        - Assess execution feasibility and time sensitivity
        - Consider flash loan opportunities for capital efficiency
        - Evaluate risk factors including slippage and front-running
        
        Provide specific execution paths and quantitative profit calculations.
        """
        
        try:
            # Generate structured analysis
            opportunities = await self.gemini_client.generate_structured_content(
                prompt=prompt,
                output_schema=List[ArbitrageOpportunity],
                config=ModelConfig(temperature=0.1, max_output_tokens=4096),
                cache_key="arbitrage_analysis"
            )
            
            # Filter by profitability and confidence
            profitable_opportunities = [
                opp for opp in opportunities
                if (opp.potential_profit >= min_profit_threshold and
                    opp.gas_cost_estimate <= max_gas_cost and
                    opp.confidence >= self.min_confidence)
            ]
            
            # Store opportunities in history
            self.pattern_history[PatternType.ARBITRAGE_OPPORTUNITY].extend(profitable_opportunities)
            
            logger.info(
                "Arbitrage opportunity detection completed",
                total_opportunities=len(opportunities),
                profitable_opportunities=len(profitable_opportunities)
            )
            
            return profitable_opportunities
            
        except Exception as e:
            logger.error("Arbitrage opportunity detection failed", error=str(e))
            raise
    
    async def analyze_market_correlations(
        self,
        pool_data_series: Dict[str, List[PoolData]],
        correlation_threshold: float = 0.7,
        min_data_points: int = 50
    ) -> List[MarketCorrelation]:
        """
        Analyze market correlations between assets
        
        Args:
            pool_data_series: Time series data for multiple pools
            correlation_threshold: Minimum correlation coefficient
            min_data_points: Minimum data points for reliable correlation
            
        Returns:
            List of detected market correlations
        """
        logger.info(
            "Starting market correlation analysis",
            pool_count=len(pool_data_series),
            correlation_threshold=correlation_threshold,
            min_data_points=min_data_points
        )
        
        # Filter pools with sufficient data
        valid_pools = {
            pool_id: data for pool_id, data in pool_data_series.items()
            if len(data) >= min_data_points
        }
        
        if len(valid_pools) < 2:
            logger.warning("Insufficient data for correlation analysis")
            return []
        
        # Prepare correlation analysis data
        correlation_data = self._prepare_correlation_analysis_data(valid_pools)
        
        # Create analysis prompt
        prompt = f"""
        Analyze market correlations between DeFi assets using the following data:
        
        Asset Price and Volume Correlation Data:
        {json.dumps(correlation_data, indent=2)}
        
        Correlation Analysis Requirements:
        - Minimum correlation threshold: {correlation_threshold}
        - Statistical significance testing (p-value < 0.05)
        - Consider both price and volume correlations
        - Identify leading and lagging indicators
        - Assess correlation stability over time
        
        Analysis Focus:
        - Calculate Pearson correlation coefficients
        - Identify time-lagged correlations
        - Detect correlation breakdowns during market stress
        - Consider fundamental reasons for correlations
        - Assess predictive value for trading strategies
        
        Provide statistical significance metrics and temporal analysis.
        """
        
        try:
            # Generate structured analysis
            correlations = await self.gemini_client.generate_structured_content(
                prompt=prompt,
                output_schema=List[MarketCorrelation],
                config=ModelConfig(temperature=0.2, max_output_tokens=4096),
                cache_key="correlation_analysis"
            )
            
            # Filter by correlation strength and confidence
            strong_correlations = [
                corr for corr in correlations
                if (abs(corr.correlation_strength) >= correlation_threshold and
                    corr.confidence >= self.min_confidence and
                    corr.statistical_significance <= 0.05)
            ]
            
            # Store correlations in history
            self.pattern_history[PatternType.MARKET_CORRELATION].extend(strong_correlations)
            
            logger.info(
                "Market correlation analysis completed",
                total_correlations=len(correlations),
                strong_correlations=len(strong_correlations)
            )
            
            return strong_correlations
            
        except Exception as e:
            logger.error("Market correlation analysis failed", error=str(e))
            raise
    
    def _prepare_pool_data_summary(self, pool_data: List[PoolData]) -> Dict[str, Any]:
        """Prepare pool data summary for analysis"""
        if not pool_data:
            return {}
        
        # Convert to DataFrame for analysis
        df_data = [pool.to_dict() for pool in pool_data]
        df = pd.DataFrame(df_data)
        
        return {
            "total_pools": len(df),
            "unique_tokens": len(set(df['token0_symbol'].tolist() + df['token1_symbol'].tolist())),
            "total_volume_24h": df['volume_24h'].sum(),
            "total_liquidity": df['liquidity'].sum(),
            "avg_price_volatility": df['price'].std() / df['price'].mean() if len(df) > 1 else 0,
            "top_volume_pools": df.nlargest(10, 'volume_24h')[['pool_address', 'token0_symbol', 'token1_symbol', 'volume_24h']].to_dict('records'),
            "price_ranges": {
                token: {
                    "min": df[df['token0_symbol'] == token]['price'].min(),
                    "max": df[df['token0_symbol'] == token]['price'].max(),
                    "current": df[df['token0_symbol'] == token]['price'].iloc[-1] if len(df[df['token0_symbol'] == token]) > 0 else 0
                }
                for token in df['token0_symbol'].unique()[:20]  # Limit to top 20 tokens
            },
            "time_range": {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max()
            }
        }
    
    def _prepare_voting_data_summary(self, voting_data: List[VotingData]) -> Dict[str, Any]:
        """Prepare voting data summary for analysis"""
        if not voting_data:
            return {}
        
        # Convert to DataFrame for analysis
        df_data = [vote.to_dict() for vote in voting_data]
        df = pd.DataFrame(df_data)
        
        return {
            "total_votes": len(df),
            "unique_voters": df['voter_address'].nunique(),
            "unique_proposals": df['proposal_id'].nunique(),
            "total_voting_power": df['voting_power'].sum(),
            "vote_distribution": df['vote_choice'].value_counts().to_dict(),
            "top_voters": df.groupby('voter_address')['voting_power'].sum().nlargest(20).to_dict(),
            "proposal_participation": df.groupby('proposal_id').agg({
                'voter_address': 'count',
                'voting_power': 'sum',
                'vote_choice': lambda x: x.value_counts().to_dict()
            }).to_dict('index'),
            "temporal_patterns": {
                "votes_per_day": df.groupby(df['timestamp'].str[:10])['voter_address'].count().to_dict(),
                "avg_voting_power_trend": df.groupby(df['timestamp'].str[:10])['voting_power'].mean().to_dict()
            },
            "time_range": {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max()
            }
        }
    
    def _group_pools_by_token_pairs(self, pool_data: List[PoolData]) -> Dict[str, List[PoolData]]:
        """Group pools by token pairs for arbitrage analysis"""
        token_pairs = {}
        
        for pool in pool_data:
            # Create normalized token pair key
            tokens = sorted([pool.token0_symbol, pool.token1_symbol])
            pair_key = f"{tokens[0]}/{tokens[1]}"
            
            if pair_key not in token_pairs:
                token_pairs[pair_key] = []
            
            token_pairs[pair_key].append(pool)
        
        # Only return pairs with multiple pools
        return {pair: pools for pair, pools in token_pairs.items() if len(pools) > 1}
    
    def _prepare_arbitrage_analysis_data(self, token_pair_pools: Dict[str, List[PoolData]]) -> Dict[str, Any]:
        """Prepare arbitrage analysis data"""
        arbitrage_data = {}
        
        for pair, pools in token_pair_pools.items():
            if len(pools) < 2:
                continue
            
            pool_info = []
            for pool in pools:
                pool_info.append({
                    "pool_address": pool.pool_address,
                    "price": pool.price,
                    "liquidity": pool.liquidity,
                    "volume_24h": pool.volume_24h,
                    "fees_24h": pool.fees_24h
                })
            
            # Calculate price differences
            prices = [p["price"] for p in pool_info]
            max_price = max(prices)
            min_price = min(prices)
            price_diff_pct = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0
            
            arbitrage_data[pair] = {
                "pools": pool_info,
                "price_difference_pct": price_diff_pct,
                "max_price": max_price,
                "min_price": min_price,
                "total_liquidity": sum(p["liquidity"] for p in pool_info),
                "total_volume": sum(p["volume_24h"] for p in pool_info)
            }
        
        return arbitrage_data
    
    def _prepare_correlation_analysis_data(self, pool_data_series: Dict[str, List[PoolData]]) -> Dict[str, Any]:
        """Prepare correlation analysis data"""
        correlation_data = {}
        
        for pool_id, data_series in pool_data_series.items():
            if len(data_series) < 10:  # Need minimum data points
                continue
            
            prices = [pool.price for pool in data_series]
            volumes = [pool.volume_24h for pool in data_series]
            timestamps = [pool.timestamp.isoformat() for pool in data_series]
            
            correlation_data[pool_id] = {
                "token_pair": f"{data_series[0].token0_symbol}/{data_series[0].token1_symbol}",
                "price_series": prices,
                "volume_series": volumes,
                "timestamps": timestamps,
                "data_points": len(data_series),
                "price_volatility": np.std(prices) / np.mean(prices) if prices else 0,
                "volume_volatility": np.std(volumes) / np.mean(volumes) if volumes else 0
            }
        
        return correlation_data
    
    async def get_pattern_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of detected patterns"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        summary = {}
        
        for pattern_type in PatternType:
            recent_patterns = [
                pattern for pattern in self.pattern_history[pattern_type]
                if hasattr(pattern, 'timestamp') and pattern.timestamp >= cutoff_time
            ]
            
            summary[pattern_type.value] = {
                "count": len(recent_patterns),
                "avg_confidence": np.mean([p.confidence for p in recent_patterns]) if recent_patterns else 0,
                "high_confidence_count": len([p for p in recent_patterns if p.confidence > 0.8])
            }
        
        return summary
    
    async def export_patterns_to_cache(self, cache_key: str) -> bool:
        """Export detected patterns to Gemini cache for future analysis"""
        try:
            pattern_export = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "patterns": {}
            }
            
            for pattern_type in PatternType:
                patterns = self.pattern_history[pattern_type][-100:]  # Last 100 patterns
                pattern_export["patterns"][pattern_type.value] = [
                    pattern.dict() if hasattr(pattern, 'dict') else str(pattern)
                    for pattern in patterns
                ]
            
            export_content = [json.dumps(pattern_export, indent=2)]
            
            cache_name = await self.gemini_client.create_cache(
                cache_key=cache_key,
                content=export_content,
                ttl_hours=48
            )
            
            if cache_name:
                logger.info("Patterns exported to cache", cache_key=cache_key, cache_name=cache_name)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error("Failed to export patterns to cache", error=str(e))
            return False