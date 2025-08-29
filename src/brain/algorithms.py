"""
Advanced algorithms for opportunity scoring and risk assessment.

This module implements sophisticated financial algorithms for:
- Multi-factor opportunity scoring
- Comprehensive risk assessment
- Portfolio optimization
- Performance measurement
- Market analysis

Uses quantitative finance principles and machine learning techniques.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import optimize
from collections import defaultdict

from .state import BrainConfig, OpportunityScore, RiskAssessment


@dataclass
class MarketMetrics:
    """Market metrics for analysis."""
    volatility: float
    volume_trend: float
    liquidity_depth: float
    price_stability: float
    correlation: float


class OpportunityScorer:
    """
    Advanced opportunity scoring using multi-factor analysis.
    
    Implements sophisticated scoring algorithms including:
    - Yield analysis with risk adjustment
    - Volume and liquidity assessment
    - Historical performance weighting
    - Pattern recognition scoring
    - Temporal factor analysis
    - Risk-adjusted returns
    """

    def __init__(self, config: BrainConfig):
        """Initialize the opportunity scorer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Scoring model parameters
        self.base_weights = {
            'yield_factor': 0.25,
            'volume_factor': 0.20,
            'liquidity_factor': 0.15,
            'stability_factor': 0.15,
            'historical_factor': 0.15,
            'timing_factor': 0.10
        }
        
        # Risk-return parameters
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.market_risk_premium = 0.08  # 8% market risk premium

    def calculate_opportunity_score(
        self, 
        pool_data: Dict[str, Any],
        market_context: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        memory_patterns: List[Dict[str, Any]]
    ) -> OpportunityScore:
        """
        Calculate comprehensive opportunity score.
        
        Args:
            pool_data: Current pool data and metrics
            market_context: Overall market conditions
            historical_data: Historical performance data
            memory_patterns: Learned patterns from memory
            
        Returns:
            Detailed opportunity score breakdown
        """
        
        try:
            # Calculate individual factor scores
            yield_score = self._calculate_yield_score(pool_data, market_context)
            volume_score = self._calculate_volume_score(pool_data, market_context)
            liquidity_score = self._calculate_liquidity_score(pool_data)
            stability_score = self._calculate_stability_score(pool_data, historical_data)
            historical_score = self._calculate_historical_score(pool_data, historical_data)
            timing_score = self._calculate_timing_score(pool_data, market_context)
            
            # Apply dynamic weighting based on market conditions
            dynamic_weights = self._calculate_dynamic_weights(market_context)
            
            # Calculate weighted total score
            total_score = (
                yield_score * dynamic_weights['yield_factor'] +
                volume_score * dynamic_weights['volume_factor'] +
                liquidity_score * dynamic_weights['liquidity_factor'] +
                stability_score * dynamic_weights['stability_factor'] +
                historical_score * dynamic_weights['historical_factor'] +
                timing_score * dynamic_weights['timing_factor']
            )
            
            # Apply pattern matching boost
            pattern_score = self._calculate_pattern_score(pool_data, memory_patterns)
            total_score = total_score * (1 + pattern_score * 0.2)  # Up to 20% boost
            
            # Calculate confidence in the score
            confidence = self._calculate_scoring_confidence(
                pool_data, market_context, historical_data
            )
            
            # Risk adjustment
            risk_adjusted_score = self._apply_risk_adjustment(total_score, pool_data)
            
            return {
                'yield_score': yield_score,
                'volume_score': volume_score,
                'tvl_score': liquidity_score,
                'historical_score': historical_score,
                'pattern_score': pattern_score,
                'timing_score': timing_score,
                'total_score': min(risk_adjusted_score, 1.0),
                'confidence': confidence,
                'factors_considered': list(dynamic_weights.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {e}")
            
            # Return neutral score on error
            return {
                'yield_score': 0.5,
                'volume_score': 0.5,
                'tvl_score': 0.5,
                'historical_score': 0.5,
                'pattern_score': 0.5,
                'timing_score': 0.5,
                'total_score': 0.5,
                'confidence': 0.0,
                'factors_considered': []
            }

    def _calculate_yield_score(
        self, 
        pool_data: Dict[str, Any], 
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate yield score with risk adjustment."""
        
        # Base APR/APY
        base_apr = pool_data.get('apr', 0)
        
        # Add emissions and rewards
        emissions_apr = pool_data.get('emissions_apr', 0)
        fee_apr = pool_data.get('fee_apr', 0)
        
        total_yield = base_apr + emissions_apr + fee_apr
        
        if total_yield <= 0:
            return 0.0
        
        # Risk adjustment using Sharpe ratio concept
        pool_volatility = pool_data.get('volatility', 0.3)
        risk_adjusted_yield = (total_yield - self.risk_free_rate) / (1 + pool_volatility)
        
        # Sigmoid normalization (50% APR = 0.8 score)
        normalized_score = 2 / (1 + np.exp(-risk_adjusted_yield / 25)) - 1
        
        # Apply sustainability factor
        sustainability_factor = self._assess_yield_sustainability(pool_data)
        
        return normalized_score * sustainability_factor

    def _calculate_volume_score(
        self, 
        pool_data: Dict[str, Any], 
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate volume score with trend analysis."""
        
        volume_24h = pool_data.get('volume_24h', 0)
        volume_7d = pool_data.get('volume_7d', 0)
        volume_30d = pool_data.get('volume_30d', 0)
        
        if volume_24h <= 0:
            return 0.0
        
        # Volume trend analysis
        volume_trend = self._calculate_volume_trend(volume_24h, volume_7d, volume_30d)
        
        # Relative volume compared to TVL
        tvl = pool_data.get('tvl', 1)
        volume_to_tvl_ratio = volume_24h / max(tvl, 1)
        
        # Healthy ratio is 1-5% of TVL per day
        volume_efficiency = min(volume_to_tvl_ratio / 0.03, 1.0)
        
        # Market context adjustment
        market_volume_factor = market_context.get('volume_trend_factor', 1.0)
        
        # Combined score
        volume_score = (volume_trend * 0.4 + volume_efficiency * 0.6) * market_volume_factor
        
        return min(volume_score, 1.0)

    def _calculate_liquidity_score(self, pool_data: Dict[str, Any]) -> float:
        """Calculate liquidity score based on depth and stability."""
        
        tvl = pool_data.get('tvl', 0)
        reserves = pool_data.get('reserves', {})
        
        if tvl <= 0:
            return 0.0
        
        # Base liquidity score (logarithmic scaling)
        base_liquidity = np.log10(max(tvl, 1)) / np.log10(10_000_000)  # Normalize to $10M
        base_liquidity = min(base_liquidity, 1.0)
        
        # Reserve balance (for AMM pools)
        reserve_balance = self._calculate_reserve_balance(reserves)
        
        # Liquidity depth (resistance to large trades)
        depth_score = self._calculate_liquidity_depth_score(pool_data)
        
        # Combine factors
        liquidity_score = (base_liquidity * 0.5 + reserve_balance * 0.3 + depth_score * 0.2)
        
        return liquidity_score

    def _calculate_stability_score(
        self, 
        pool_data: Dict[str, Any], 
        historical_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate stability score based on historical volatility."""
        
        if not historical_data:
            # Default stability for stable pairs
            is_stable = pool_data.get('is_stable', False)
            return 0.8 if is_stable else 0.5
        
        # Calculate price volatility
        price_volatility = self._calculate_price_volatility(historical_data)
        
        # Calculate TVL volatility
        tvl_volatility = self._calculate_tvl_volatility(historical_data)
        
        # Calculate volume volatility
        volume_volatility = self._calculate_volume_volatility(historical_data)
        
        # Combined stability (lower volatility = higher stability)
        stability_score = (
            (1 - min(price_volatility, 1)) * 0.5 +
            (1 - min(tvl_volatility, 1)) * 0.3 +
            (1 - min(volume_volatility, 1)) * 0.2
        )
        
        return stability_score

    def _calculate_historical_score(
        self, 
        pool_data: Dict[str, Any], 
        historical_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate score based on historical performance."""
        
        if not historical_data:
            return 0.5  # Neutral for new pools
        
        # Calculate historical returns
        returns = self._extract_historical_returns(historical_data)
        
        if not returns:
            return 0.5
        
        # Sharpe ratio calculation
        mean_return = np.mean(returns)
        return_volatility = np.std(returns)
        
        if return_volatility == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (mean_return - self.risk_free_rate / 252) / return_volatility  # Daily
        
        # Normalize Sharpe ratio to 0-1 score
        sharpe_score = (sharpe_ratio + 2) / 4  # Sharpe of 2 = score of 1
        sharpe_score = max(0, min(sharpe_score, 1))
        
        # Maximum drawdown penalty
        max_drawdown = self._calculate_max_drawdown(returns)
        drawdown_penalty = min(max_drawdown * 2, 0.5)  # Up to 50% penalty
        
        # Win rate
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        
        # Combined historical score
        historical_score = (sharpe_score * 0.5 + win_rate * 0.3) * (1 - drawdown_penalty)
        
        return max(historical_score, 0)

    def _calculate_timing_score(
        self, 
        pool_data: Dict[str, Any], 
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate timing score based on market conditions."""
        
        # Gas price factor
        gas_price = market_context.get('gas_price', 0.001)
        gas_score = max(0, 1 - (gas_price / 0.05))  # Good if under 0.05 gwei
        
        # Network congestion factor
        congestion = market_context.get('network_congestion', 0.5)
        congestion_score = 1 - congestion
        
        # Market sentiment factor
        sentiment = market_context.get('market_sentiment', {})
        sentiment_score = self._score_market_sentiment(sentiment)
        
        # Volatility timing (prefer lower volatility for LP, higher for trading)
        volatility = pool_data.get('volatility', 0.3)
        volatility_score = self._score_volatility_timing(volatility, pool_data)
        
        # Time-of-day factor
        current_hour = datetime.now().hour
        time_score = self._score_time_of_day(current_hour)
        
        # Combined timing score
        timing_score = (
            gas_score * 0.3 +
            congestion_score * 0.25 +
            sentiment_score * 0.2 +
            volatility_score * 0.15 +
            time_score * 0.1
        )
        
        return timing_score

    def _calculate_pattern_score(
        self, 
        pool_data: Dict[str, Any], 
        memory_patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate pattern matching score."""
        
        if not memory_patterns:
            return 0.5  # Neutral if no patterns
        
        pool_features = self._extract_pool_features(pool_data)
        
        pattern_matches = []
        
        for pattern in memory_patterns:
            pattern_features = pattern.get('pattern_features', {})
            
            # Calculate feature similarity
            similarity = self._calculate_feature_similarity(pool_features, pattern_features)
            
            # Weight by pattern success rate and confidence
            pattern_weight = (
                pattern.get('success_rate', 0.5) * 0.7 +
                pattern.get('confidence_level', 0.5) * 0.3
            )
            
            weighted_similarity = similarity * pattern_weight
            pattern_matches.append(weighted_similarity)
        
        if not pattern_matches:
            return 0.5
        
        # Use weighted average of top 3 patterns
        top_matches = sorted(pattern_matches, reverse=True)[:3]
        
        return np.mean(top_matches)

    def _calculate_dynamic_weights(self, market_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic weights based on market conditions."""
        
        weights = self.base_weights.copy()
        
        # Adjust weights based on market volatility
        market_volatility = market_context.get('volatility', 0.3)
        
        if market_volatility > 0.5:  # High volatility
            # Emphasize stability and reduce yield weight
            weights['stability_factor'] *= 1.3
            weights['yield_factor'] *= 0.8
        
        # Adjust based on gas prices
        gas_price = market_context.get('gas_price', 0.001)
        
        if gas_price > 0.02:  # High gas prices
            # Emphasize timing
            weights['timing_factor'] *= 1.5
            weights['yield_factor'] *= 0.9
        
        # Normalize weights
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        return weights

    def _calculate_scoring_confidence(
        self, 
        pool_data: Dict[str, Any], 
        market_context: Dict[str, Any], 
        historical_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in the scoring."""
        
        confidence_factors = []
        
        # Data completeness
        required_fields = ['apr', 'tvl', 'volume_24h', 'reserves']
        completeness = sum(1 for field in required_fields if pool_data.get(field) is not None)
        data_confidence = completeness / len(required_fields)
        confidence_factors.append(data_confidence)
        
        # Historical data availability
        if historical_data:
            history_confidence = min(len(historical_data) / 30, 1.0)  # 30 days = full confidence
        else:
            history_confidence = 0.3  # Low confidence without history
        
        confidence_factors.append(history_confidence)
        
        # Market data quality
        market_data_quality = market_context.get('data_quality', 0.8)
        confidence_factors.append(market_data_quality)
        
        # Pool maturity (older pools = more confidence)
        pool_age_days = pool_data.get('age_days', 0)
        maturity_confidence = min(pool_age_days / 90, 1.0)  # 90 days = full confidence
        confidence_factors.append(maturity_confidence)
        
        return np.mean(confidence_factors)

    def _apply_risk_adjustment(self, score: float, pool_data: Dict[str, Any]) -> float:
        """Apply risk adjustment to the base score."""
        
        # Calculate pool-specific risk factors
        volatility_risk = min(pool_data.get('volatility', 0.3) / 0.5, 1.0)
        liquidity_risk = max(0, 1 - (pool_data.get('tvl', 0) / 100_000))
        smart_contract_risk = pool_data.get('contract_risk_score', 0.2)
        
        # Combined risk factor
        total_risk = (volatility_risk + liquidity_risk + smart_contract_risk) / 3
        
        # Risk adjustment (higher risk = lower adjustment)
        risk_adjustment = 1 - (total_risk * 0.3)  # Max 30% reduction
        
        return score * risk_adjustment

    # Helper methods for calculations

    def _assess_yield_sustainability(self, pool_data: Dict[str, Any]) -> float:
        """Assess if the current yield is sustainable."""
        
        apr = pool_data.get('apr', 0)
        
        # Very high APRs (>200%) are often unsustainable
        if apr > 200:
            return 0.3
        elif apr > 100:
            return 0.6
        elif apr > 50:
            return 0.8
        else:
            return 1.0

    def _calculate_volume_trend(
        self, 
        volume_24h: float, 
        volume_7d: float, 
        volume_30d: float
    ) -> float:
        """Calculate volume trend score."""
        
        if volume_7d <= 0 or volume_30d <= 0:
            return 0.5  # Neutral if no historical data
        
        # Short-term trend (24h vs 7d average)
        volume_7d_avg = volume_7d / 7
        short_trend = volume_24h / volume_7d_avg if volume_7d_avg > 0 else 1
        
        # Long-term trend (7d vs 30d average)
        volume_30d_avg = volume_30d / 30
        long_trend = volume_7d_avg / volume_30d_avg if volume_30d_avg > 0 else 1
        
        # Combine trends (prefer growing volume)
        trend_score = (
            min(short_trend / 2, 1.0) * 0.6 +  # Short-term weight
            min(long_trend / 2, 1.0) * 0.4     # Long-term weight
        )
        
        return trend_score

    def _calculate_reserve_balance(self, reserves: Dict[str, Any]) -> float:
        """Calculate reserve balance for AMM pools."""
        
        if not reserves:
            return 1.0  # Assume balanced if no data
        
        reserve0 = reserves.get('reserve0', 0)
        reserve1 = reserves.get('reserve1', 0)
        
        if reserve0 <= 0 or reserve1 <= 0:
            return 0.0
        
        # Calculate balance (closer to 50/50 = better)
        ratio = min(reserve0, reserve1) / max(reserve0, reserve1)
        
        return ratio

    def _calculate_liquidity_depth_score(self, pool_data: Dict[str, Any]) -> float:
        """Calculate liquidity depth score."""
        
        # This would analyze price impact for different trade sizes
        # Placeholder implementation
        
        tvl = pool_data.get('tvl', 0)
        volume_24h = pool_data.get('volume_24h', 0)
        
        if tvl <= 0:
            return 0.0
        
        # Higher TVL relative to volume = better depth
        depth_ratio = tvl / max(volume_24h, tvl * 0.01)  # At least 1% daily volume
        
        return min(depth_ratio / 50, 1.0)  # Normalize

    def _calculate_price_volatility(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate price volatility from historical data."""
        
        prices = [data.get('price', 0) for data in historical_data if data.get('price')]
        
        if len(prices) < 2:
            return 0.5  # Default moderate volatility
        
        # Calculate daily returns
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices)) if prices[i-1] > 0]
        
        if not returns:
            return 0.5
        
        # Standard deviation of returns (annualized)
        daily_volatility = np.std(returns)
        annual_volatility = daily_volatility * np.sqrt(252)  # 252 trading days
        
        return min(annual_volatility, 2.0)  # Cap at 200% volatility

    def _calculate_tvl_volatility(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate TVL volatility."""
        
        tvls = [data.get('tvl', 0) for data in historical_data if data.get('tvl')]
        
        if len(tvls) < 2:
            return 0.3  # Default low TVL volatility
        
        # Calculate TVL changes
        changes = [(tvls[i] / tvls[i-1] - 1) for i in range(1, len(tvls)) if tvls[i-1] > 0]
        
        if not changes:
            return 0.3
        
        return min(np.std(changes), 1.0)

    def _calculate_volume_volatility(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate volume volatility."""
        
        volumes = [data.get('volume_24h', 0) for data in historical_data if data.get('volume_24h')]
        
        if len(volumes) < 2:
            return 0.5  # Default moderate volume volatility
        
        # Coefficient of variation for volume
        mean_volume = np.mean(volumes)
        if mean_volume <= 0:
            return 0.5
        
        cv = np.std(volumes) / mean_volume
        
        return min(cv, 2.0)

    def _extract_historical_returns(self, historical_data: List[Dict[str, Any]]) -> List[float]:
        """Extract returns from historical data."""
        
        # This would calculate actual LP returns including fees and IL
        # Placeholder implementation
        
        returns = []
        for i, data in enumerate(historical_data):
            if i == 0:
                continue
                
            prev_data = historical_data[i-1]
            
            # Simple price-based return (would be more complex for LP positions)
            price = data.get('price', 0)
            prev_price = prev_data.get('price', 0)
            
            if prev_price > 0:
                daily_return = (price / prev_price) - 1
                returns.append(daily_return)
        
        return returns

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        
        if not returns:
            return 0.0
        
        cumulative_returns = np.cumprod(1 + np.array(returns))
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        
        return abs(np.min(drawdown))

    def _score_market_sentiment(self, sentiment: Dict[str, Any]) -> float:
        """Score market sentiment for timing."""
        
        sentiment_type = sentiment.get('sentiment', 'neutral')
        confidence = sentiment.get('confidence', 0.5)
        
        sentiment_scores = {
            'very_bullish': 0.9,
            'bullish': 0.8,
            'active': 0.7,
            'neutral': 0.5,
            'bearish': 0.3,
            'very_bearish': 0.1
        }
        
        base_score = sentiment_scores.get(sentiment_type, 0.5)
        
        # Weight by confidence
        return base_score * confidence + 0.5 * (1 - confidence)

    def _score_volatility_timing(self, volatility: float, pool_data: Dict[str, Any]) -> float:
        """Score volatility timing based on strategy."""
        
        # For LP positions, prefer lower volatility
        # For trading, might prefer higher volatility
        
        is_stable = pool_data.get('is_stable', False)
        
        if is_stable:
            # Stable pairs: prefer very low volatility
            return max(0, 1 - (volatility / 0.1))
        else:
            # Volatile pairs: moderate volatility is good, extreme is bad
            if volatility < 0.1:
                return 0.6  # Too stable for volatile pair
            elif volatility < 0.5:
                return 0.9  # Good volatility
            elif volatility < 1.0:
                return 0.7  # High but manageable
            else:
                return 0.3  # Too volatile

    def _score_time_of_day(self, hour: int) -> float:
        """Score based on time of day (market activity patterns)."""
        
        # DeFi is 24/7, but some times have higher activity
        # This is a simplified model
        
        if 8 <= hour <= 16:  # Business hours (various timezones)
            return 0.8
        elif 0 <= hour <= 6:  # Low activity
            return 0.6
        else:  # Evening hours
            return 0.7

    def _extract_pool_features(self, pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for pattern matching."""
        
        return {
            'is_stable': pool_data.get('is_stable', False),
            'tvl_range': self._categorize_value(pool_data.get('tvl', 0), [100000, 1000000, 10000000]),
            'apr_range': self._categorize_value(pool_data.get('apr', 0), [10, 50, 100]),
            'volume_range': self._categorize_value(pool_data.get('volume_24h', 0), [10000, 100000, 1000000]),
            'volatility_range': self._categorize_value(pool_data.get('volatility', 0), [0.1, 0.3, 0.5])
        }

    def _calculate_feature_similarity(
        self, 
        features1: Dict[str, Any], 
        features2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between feature sets."""
        
        matches = 0
        total = 0
        
        for key in features1:
            if key in features2:
                total += 1
                if features1[key] == features2[key]:
                    matches += 1
        
        return matches / total if total > 0 else 0

    def _categorize_value(self, value: float, thresholds: List[float]) -> str:
        """Categorize a value based on thresholds."""
        
        categories = ['very_low', 'low', 'medium', 'high', 'very_high']
        
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return categories[i]
        
        return categories[-1]


class RiskAssessor:
    """
    Advanced risk assessment using multiple risk models.
    
    Implements comprehensive risk analysis including:
    - Value at Risk (VaR) calculations
    - Conditional VaR (CVaR)
    - Portfolio risk metrics
    - Concentration risk analysis
    - Market risk assessment
    - Operational risk evaluation
    """

    def __init__(self, config: BrainConfig):
        """Initialize the risk assessor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk model parameters
        self.confidence_levels = [0.95, 0.99]
        self.time_horizons = [1, 7, 30]  # days
        
        # Risk factor weights
        self.risk_weights = {
            'market_risk': 0.25,
            'liquidity_risk': 0.20,
            'concentration_risk': 0.20,
            'operational_risk': 0.15,
            'technical_risk': 0.10,
            'regulatory_risk': 0.10
        }

    def assess_risk(
        self, 
        position_data: Dict[str, Any],
        portfolio_context: Dict[str, Any],
        market_conditions: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.
        
        Args:
            position_data: Data about the proposed position
            portfolio_context: Current portfolio state
            market_conditions: Market environment data
            historical_data: Historical performance data
            
        Returns:
            Detailed risk assessment
        """
        
        try:
            # Calculate individual risk components
            market_risk = self._calculate_market_risk(position_data, market_conditions, historical_data)
            liquidity_risk = self._calculate_liquidity_risk(position_data, market_conditions)
            concentration_risk = self._calculate_concentration_risk(position_data, portfolio_context)
            operational_risk = self._calculate_operational_risk(position_data, market_conditions)
            technical_risk = self._calculate_technical_risk(position_data)
            regulatory_risk = self._calculate_regulatory_risk(position_data, market_conditions)
            
            # Calculate composite risk score
            total_risk = (
                market_risk * self.risk_weights['market_risk'] +
                liquidity_risk * self.risk_weights['liquidity_risk'] +
                concentration_risk * self.risk_weights['concentration_risk'] +
                operational_risk * self.risk_weights['operational_risk'] +
                technical_risk * self.risk_weights['technical_risk'] +
                regulatory_risk * self.risk_weights['regulatory_risk']
            )
            
            # Determine risk acceptability
            acceptable = total_risk <= self.config.risk_threshold
            
            # Generate recommendation
            if total_risk <= 0.3:
                recommendation = "proceed"
            elif total_risk <= 0.6:
                recommendation = "caution"
            else:
                recommendation = "abort"
            
            # Identify primary risk factors
            risk_factors = self._identify_primary_risk_factors({
                'market_risk': market_risk,
                'liquidity_risk': liquidity_risk,
                'concentration_risk': concentration_risk,
                'operational_risk': operational_risk,
                'technical_risk': technical_risk,
                'regulatory_risk': regulatory_risk
            })
            
            # Generate mitigation strategies
            mitigations = self._generate_risk_mitigations(risk_factors, position_data)
            
            return {
                'position_risk': concentration_risk,
                'market_risk': market_risk,
                'liquidity_risk': liquidity_risk,
                'contract_risk': technical_risk,
                'timing_risk': operational_risk,
                'impermanent_loss_risk': self._calculate_il_risk(position_data),
                'total_risk': total_risk,
                'acceptable': acceptable,
                'recommendation': recommendation,
                'risk_factors': risk_factors,
                'mitigations': mitigations
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            
            # Return high-risk assessment on error
            return {
                'position_risk': 1.0,
                'market_risk': 1.0,
                'liquidity_risk': 1.0,
                'contract_risk': 1.0,
                'timing_risk': 1.0,
                'impermanent_loss_risk': 1.0,
                'total_risk': 1.0,
                'acceptable': False,
                'recommendation': "abort",
                'risk_factors': ["assessment_error"],
                'mitigations': ["retry_assessment"]
            }

    def _calculate_market_risk(
        self, 
        position_data: Dict[str, Any],
        market_conditions: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Calculate market risk using various models."""
        
        # Price volatility risk
        volatility = position_data.get('volatility', 0.3)
        volatility_risk = min(volatility / 2.0, 1.0)  # Normalize to 200% volatility
        
        # Market correlation risk
        correlation_risk = self._calculate_correlation_risk(position_data, market_conditions)
        
        # Beta risk (systematic risk)
        beta_risk = self._calculate_beta_risk(position_data, market_conditions, historical_data)
        
        # VaR calculation
        var_risk = self._calculate_var(position_data, historical_data)
        
        # Combined market risk
        market_risk = (
            volatility_risk * 0.3 +
            correlation_risk * 0.25 +
            beta_risk * 0.25 +
            var_risk * 0.2
        )
        
        return market_risk

    def _calculate_liquidity_risk(
        self, 
        position_data: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate liquidity risk."""
        
        tvl = position_data.get('tvl', 0)
        volume_24h = position_data.get('volume_24h', 0)
        position_size = position_data.get('position_size_usd', 0)
        
        # Size impact risk
        if tvl > 0:
            size_impact = min(position_size / tvl, 1.0)
        else:
            size_impact = 1.0
        
        # Volume adequacy risk
        daily_volume_ratio = volume_24h / max(tvl, 1)
        volume_risk = max(0, 1 - (daily_volume_ratio / 0.1))  # Should have >10% daily volume/TVL
        
        # Market liquidity conditions
        market_liquidity_risk = market_conditions.get('liquidity_stress', 0.3)
        
        # Slippage risk
        expected_slippage = position_data.get('expected_slippage', 0.01)
        slippage_risk = min(expected_slippage / 0.05, 1.0)  # Normalize to 5% slippage
        
        # Combined liquidity risk
        liquidity_risk = (
            size_impact * 0.3 +
            volume_risk * 0.25 +
            market_liquidity_risk * 0.25 +
            slippage_risk * 0.2
        )
        
        return liquidity_risk

    def _calculate_concentration_risk(
        self, 
        position_data: Dict[str, Any],
        portfolio_context: Dict[str, Any]
    ) -> float:
        """Calculate concentration risk."""
        
        position_size = position_data.get('position_size_usd', 0)
        portfolio_value = portfolio_context.get('total_value', 0)
        
        if portfolio_value <= 0:
            return 1.0  # Maximum risk if no portfolio context
        
        # Position size risk
        position_ratio = position_size / portfolio_value
        size_risk = min(position_ratio / self.config.max_position_size, 1.0)
        
        # Asset concentration risk
        existing_positions = portfolio_context.get('positions', [])
        similar_positions = self._count_similar_positions(position_data, existing_positions)
        
        # Penalize over-concentration in similar assets
        concentration_penalty = min(similar_positions / 5, 1.0)  # 5+ similar positions = max penalty
        
        # Sector/protocol concentration
        protocol_concentration = self._calculate_protocol_concentration(position_data, existing_positions)
        
        # Combined concentration risk
        concentration_risk = (
            size_risk * 0.4 +
            concentration_penalty * 0.3 +
            protocol_concentration * 0.3
        )
        
        return concentration_risk

    def _calculate_operational_risk(
        self, 
        position_data: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate operational risk."""
        
        # Gas price risk
        gas_price = market_conditions.get('gas_price', 0.001)
        gas_risk = min(gas_price / self.config.max_gas_price, 1.0)
        
        # Network congestion risk
        congestion = market_conditions.get('network_congestion', 0.5)
        
        # Execution timing risk
        execution_urgency = position_data.get('execution_urgency', 0.5)
        timing_risk = gas_risk * congestion * execution_urgency
        
        # MEV risk
        mev_risk = self._calculate_mev_risk(position_data, market_conditions)
        
        # Oracle risk
        oracle_risk = self._calculate_oracle_risk(position_data)
        
        # Combined operational risk
        operational_risk = (
            timing_risk * 0.4 +
            mev_risk * 0.3 +
            oracle_risk * 0.3
        )
        
        return operational_risk

    def _calculate_technical_risk(self, position_data: Dict[str, Any]) -> float:
        """Calculate technical/smart contract risk."""
        
        # Protocol maturity
        protocol_age_days = position_data.get('protocol_age_days', 0)
        
        if protocol_age_days < 30:
            maturity_risk = 0.8
        elif protocol_age_days < 180:
            maturity_risk = 0.5
        elif protocol_age_days < 365:
            maturity_risk = 0.3
        else:
            maturity_risk = 0.1
        
        # Audit status
        audit_risk = position_data.get('audit_risk_score', 0.3)
        
        # Code complexity
        complexity_risk = position_data.get('complexity_risk_score', 0.4)
        
        # TVL security (larger TVL = battle-tested)
        tvl = position_data.get('tvl', 0)
        tvl_security = max(0, 1 - (tvl / 10_000_000))  # Lower risk for >$10M TVL
        
        # Combined technical risk
        technical_risk = (
            maturity_risk * 0.3 +
            audit_risk * 0.3 +
            complexity_risk * 0.2 +
            tvl_security * 0.2
        )
        
        return technical_risk

    def _calculate_regulatory_risk(
        self, 
        position_data: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate regulatory risk."""
        
        # Base regulatory environment risk
        base_regulatory_risk = market_conditions.get('regulatory_risk', 0.2)
        
        # Token-specific regulatory risk
        tokens = position_data.get('tokens', [])
        token_regulatory_risk = max([self._get_token_regulatory_risk(token) for token in tokens], default=0.1)
        
        # Protocol regulatory risk
        protocol_type = position_data.get('protocol_type', 'defi')
        protocol_regulatory_risk = self._get_protocol_regulatory_risk(protocol_type)
        
        # Combined regulatory risk
        regulatory_risk = (
            base_regulatory_risk * 0.4 +
            token_regulatory_risk * 0.4 +
            protocol_regulatory_risk * 0.2
        )
        
        return regulatory_risk

    def _calculate_il_risk(self, position_data: Dict[str, Any]) -> float:
        """Calculate impermanent loss risk for LP positions."""
        
        position_type = position_data.get('position_type', 'LP')
        
        if position_type != 'LP':
            return 0.0  # No IL risk for non-LP positions
        
        is_stable = position_data.get('is_stable_pair', False)
        
        if is_stable:
            # Stable pairs have minimal IL risk
            return 0.1
        
        # For volatile pairs, IL risk depends on price correlation
        price_correlation = position_data.get('price_correlation', 0.5)
        volatility_differential = position_data.get('volatility_differential', 0.3)
        
        # Higher correlation and similar volatilities = lower IL risk
        il_risk = (1 - price_correlation) * 0.6 + volatility_differential * 0.4
        
        return min(il_risk, 1.0)

    # Helper methods for risk calculations

    def _calculate_correlation_risk(
        self, 
        position_data: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate correlation risk with market."""
        
        # Placeholder - would implement actual correlation calculation
        return 0.5

    def _calculate_beta_risk(
        self, 
        position_data: Dict[str, Any],
        market_conditions: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Calculate beta risk (systematic risk)."""
        
        # Placeholder - would implement actual beta calculation
        return 0.4

    def _calculate_var(
        self, 
        position_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Calculate Value at Risk."""
        
        if not historical_data:
            return 0.5  # Default moderate risk
        
        # Extract returns
        returns = [data.get('daily_return', 0) for data in historical_data if 'daily_return' in data]
        
        if len(returns) < 10:
            return 0.5
        
        # Calculate VaR at 95% confidence level
        var_95 = np.percentile(returns, 5)  # 5th percentile for losses
        
        # Normalize to 0-1 scale (20% loss = 1.0 risk)
        var_risk = min(abs(var_95) / 0.2, 1.0)
        
        return var_risk

    def _count_similar_positions(
        self, 
        position_data: Dict[str, Any],
        existing_positions: List[Dict[str, Any]]
    ) -> int:
        """Count similar positions in portfolio."""
        
        target_tokens = set(position_data.get('tokens', []))
        target_protocol = position_data.get('protocol', '')
        
        similar_count = 0
        
        for pos in existing_positions:
            pos_tokens = set(pos.get('tokens', []))
            pos_protocol = pos.get('protocol', '')
            
            # Check for token overlap or same protocol
            if target_tokens & pos_tokens or target_protocol == pos_protocol:
                similar_count += 1
        
        return similar_count

    def _calculate_protocol_concentration(
        self, 
        position_data: Dict[str, Any],
        existing_positions: List[Dict[str, Any]]
    ) -> float:
        """Calculate protocol concentration risk."""
        
        target_protocol = position_data.get('protocol', '')
        
        if not target_protocol:
            return 0.5
        
        protocol_exposure = 0
        total_exposure = 0
        
        for pos in existing_positions:
            pos_value = pos.get('value_usd', 0)
            total_exposure += pos_value
            
            if pos.get('protocol') == target_protocol:
                protocol_exposure += pos_value
        
        # Add current position
        position_value = position_data.get('position_size_usd', 0)
        protocol_exposure += position_value
        total_exposure += position_value
        
        if total_exposure <= 0:
            return 0
        
        concentration_ratio = protocol_exposure / total_exposure
        
        # Risk increases exponentially with concentration
        return min(concentration_ratio ** 2, 1.0)

    def _calculate_mev_risk(
        self, 
        position_data: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate MEV (Maximum Extractable Value) risk."""
        
        position_size = position_data.get('position_size_usd', 0)
        
        # Larger positions have higher MEV risk
        size_risk = min(position_size / 100_000, 1.0)  # $100k+ = high MEV risk
        
        # Market volatility affects MEV opportunity
        volatility = market_conditions.get('volatility', 0.3)
        volatility_risk = min(volatility / 0.5, 1.0)
        
        # Network congestion affects MEV extraction
        congestion = market_conditions.get('network_congestion', 0.5)
        
        mev_risk = (size_risk * 0.4 + volatility_risk * 0.3 + congestion * 0.3)
        
        return mev_risk

    def _calculate_oracle_risk(self, position_data: Dict[str, Any]) -> float:
        """Calculate oracle risk."""
        
        # Number of oracles used
        oracle_count = position_data.get('oracle_count', 1)
        redundancy_factor = max(0, 1 - (oracle_count - 1) * 0.2)  # Lower risk with more oracles
        
        # Oracle reliability
        oracle_reliability = position_data.get('oracle_reliability', 0.8)
        reliability_risk = 1 - oracle_reliability
        
        # Price feed freshness
        feed_staleness = position_data.get('price_feed_staleness_minutes', 5)
        staleness_risk = min(feed_staleness / 60, 1.0)  # >60 min = high risk
        
        oracle_risk = (redundancy_factor * 0.4 + reliability_risk * 0.4 + staleness_risk * 0.2)
        
        return oracle_risk

    def _get_token_regulatory_risk(self, token: str) -> float:
        """Get regulatory risk for specific token."""
        
        # Placeholder - would implement actual token risk assessment
        high_risk_tokens = ['USDT', 'USDC']  # Stablecoins have regulatory scrutiny
        
        if token.upper() in high_risk_tokens:
            return 0.4
        else:
            return 0.2

    def _get_protocol_regulatory_risk(self, protocol_type: str) -> float:
        """Get regulatory risk for protocol type."""
        
        risk_scores = {
            'lending': 0.3,
            'dex': 0.2,
            'derivatives': 0.5,
            'staking': 0.1,
            'yield_farming': 0.3
        }
        
        return risk_scores.get(protocol_type.lower(), 0.3)

    def _identify_primary_risk_factors(self, risk_components: Dict[str, float]) -> List[str]:
        """Identify primary risk factors."""
        
        factors = []
        threshold = 0.5  # Risk factors above 50%
        
        for risk_type, risk_value in risk_components.items():
            if risk_value > threshold:
                factors.append(risk_type.replace('_', ' ').title())
        
        return factors

    def _generate_risk_mitigations(
        self, 
        risk_factors: List[str],
        position_data: Dict[str, Any]
    ) -> List[str]:
        """Generate risk mitigation strategies."""
        
        mitigations = []
        
        for factor in risk_factors:
            if 'Market Risk' in factor:
                mitigations.append("Consider hedging strategies")
            elif 'Liquidity Risk' in factor:
                mitigations.append("Reduce position size or wait for better liquidity")
            elif 'Concentration Risk' in factor:
                mitigations.append("Diversify across different protocols/assets")
            elif 'Technical Risk' in factor:
                mitigations.append("Use more established protocols")
            elif 'Operational Risk' in factor:
                mitigations.append("Wait for better network conditions")
        
        # Add general mitigations
        if len(risk_factors) > 2:
            mitigations.append("Consider reducing overall exposure")
        
        return mitigations