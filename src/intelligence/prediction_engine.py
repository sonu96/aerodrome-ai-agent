"""
Prediction Engine

Advanced AI-powered predictive analytics system for:
- Pool performance predictions
- Voting outcome predictions  
- Price impact analysis
- Risk assessment
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from pydantic import BaseModel, Field

import structlog
from .gemini_client import GeminiClient, GeminiModel, ModelConfig, FunctionSpec, StructuredOutput
from .pattern_recognition import PoolData, VotingData, PatternRecognitionEngine

logger = structlog.get_logger(__name__)


class PredictionType(Enum):
    """Types of predictions that can be made"""
    POOL_PERFORMANCE = "pool_performance"
    VOTING_OUTCOME = "voting_outcome"
    PRICE_IMPACT = "price_impact"
    RISK_ASSESSMENT = "risk_assessment"
    LIQUIDITY_FLOW = "liquidity_flow"
    MARKET_MOVEMENT = "market_movement"


class TimeHorizon(Enum):
    """Prediction time horizons"""
    SHORT_TERM = "1h"
    MEDIUM_TERM = "24h"
    LONG_TERM = "7d"
    EXTENDED_TERM = "30d"


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class PoolPerformancePrediction(StructuredOutput):
    """Pool performance prediction"""
    pool_address: str = Field(..., description="Pool contract address")
    token_pair: str = Field(..., description="Token pair (e.g., ETH/USDC)")
    prediction_horizon: str = Field(..., description="Prediction time horizon")
    
    # Volume predictions
    predicted_volume_24h: float = Field(..., description="Predicted 24h volume in USD")
    volume_change_pct: float = Field(..., description="Expected volume change percentage")
    volume_confidence: float = Field(..., ge=0.0, le=1.0, description="Volume prediction confidence")
    
    # Price predictions
    predicted_price: float = Field(..., description="Predicted token price")
    price_change_pct: float = Field(..., description="Expected price change percentage")
    price_confidence: float = Field(..., ge=0.0, le=1.0, description="Price prediction confidence")
    
    # Liquidity predictions
    predicted_liquidity: float = Field(..., description="Predicted total liquidity in USD")
    liquidity_change_pct: float = Field(..., description="Expected liquidity change percentage")
    liquidity_confidence: float = Field(..., ge=0.0, le=1.0, description="Liquidity prediction confidence")
    
    # Fee predictions
    predicted_fees_24h: float = Field(..., description="Predicted 24h fees in USD")
    apr_estimate: float = Field(..., description="Estimated APR for liquidity providers")
    
    # Risk factors
    impermanent_loss_risk: float = Field(..., ge=0.0, le=1.0, description="Impermanent loss risk score")
    volatility_risk: float = Field(..., ge=0.0, le=1.0, description="Price volatility risk score")
    
    # Supporting factors
    key_drivers: List[str] = Field(..., description="Key factors driving the prediction")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    market_conditions: str = Field(..., description="Expected market conditions")


class VotingOutcomePrediction(StructuredOutput):
    """Voting outcome prediction"""
    proposal_id: str = Field(..., description="Proposal identifier")
    proposal_title: str = Field(..., description="Proposal title or description")
    prediction_time: str = Field(..., description="When prediction was made")
    voting_deadline: str = Field(..., description="Voting deadline")
    
    # Outcome predictions
    predicted_outcome: str = Field(..., description="Predicted outcome (pass/fail)")
    outcome_confidence: float = Field(..., ge=0.0, le=1.0, description="Outcome prediction confidence")
    
    # Vote distribution predictions
    predicted_for_votes: float = Field(..., description="Predicted votes for proposal")
    predicted_against_votes: float = Field(..., description="Predicted votes against proposal")
    predicted_abstain_votes: float = Field(..., description="Predicted abstain votes")
    
    predicted_for_pct: float = Field(..., ge=0.0, le=100.0, description="Predicted percentage voting for")
    predicted_against_pct: float = Field(..., ge=0.0, le=100.0, description="Predicted percentage voting against")
    predicted_abstain_pct: float = Field(..., ge=0.0, le=100.0, description="Predicted percentage abstaining")
    
    # Participation predictions
    predicted_turnout: float = Field(..., description="Predicted voter turnout percentage")
    predicted_voting_power: float = Field(..., description="Predicted total voting power participating")
    
    # Influential factors
    key_stakeholders: List[str] = Field(..., description="Key stakeholder addresses likely to influence outcome")
    coalition_impact: Dict[str, float] = Field(..., description="Impact of detected coalitions")
    sentiment_factors: List[str] = Field(..., description="Factors influencing voter sentiment")
    
    # Risk assessment
    uncertainty_factors: List[str] = Field(..., description="Factors that could change the outcome")
    last_minute_shift_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of last-minute outcome shift")


class PriceImpactAnalysis(StructuredOutput):
    """Price impact analysis"""
    pool_address: str = Field(..., description="Pool contract address")
    token_pair: str = Field(..., description="Token pair being analyzed")
    trade_scenario: str = Field(..., description="Trade scenario description")
    
    # Trade details
    trade_size_usd: float = Field(..., description="Trade size in USD")
    trade_direction: str = Field(..., description="Trade direction (buy/sell)")
    
    # Impact predictions
    price_impact_pct: float = Field(..., description="Predicted price impact percentage")
    slippage_estimate: float = Field(..., description="Estimated slippage percentage")
    
    # Liquidity analysis
    available_liquidity: float = Field(..., description="Available liquidity in trade direction")
    liquidity_utilization_pct: float = Field(..., description="Percentage of liquidity utilized")
    
    # Market depth
    market_depth_5_pct: float = Field(..., description="Market depth for 5% price impact")
    market_depth_10_pct: float = Field(..., description="Market depth for 10% price impact")
    
    # Recovery analysis
    price_recovery_time: str = Field(..., description="Estimated time for price recovery")
    recovery_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of full price recovery")
    
    # MEV considerations
    mev_risk_score: float = Field(..., ge=0.0, le=1.0, description="MEV (front-running/sandwich) risk score")
    optimal_execution_strategy: str = Field(..., description="Recommended execution strategy")
    
    # Alternative routes
    alternative_routes: List[Dict[str, Any]] = Field(..., description="Alternative trading routes with better impact")


class RiskAssessment(StructuredOutput):
    """Comprehensive risk assessment"""
    assessment_type: str = Field(..., description="Type of risk assessment")
    subject_identifier: str = Field(..., description="Subject being assessed (pool, proposal, etc.)")
    assessment_timeframe: str = Field(..., description="Assessment time horizon")
    
    # Overall risk metrics
    overall_risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    risk_level: str = Field(..., description="Risk level (low/medium/high/extreme)")
    
    # Specific risk categories
    market_risk: float = Field(..., ge=0.0, le=1.0, description="Market/price risk score")
    liquidity_risk: float = Field(..., ge=0.0, le=1.0, description="Liquidity risk score")
    operational_risk: float = Field(..., ge=0.0, le=1.0, description="Operational/smart contract risk score")
    regulatory_risk: float = Field(..., ge=0.0, le=1.0, description="Regulatory risk score")
    counterparty_risk: float = Field(..., ge=0.0, le=1.0, description="Counterparty risk score")
    
    # Risk factors
    high_risk_factors: List[str] = Field(..., description="High-priority risk factors")
    medium_risk_factors: List[str] = Field(..., description="Medium-priority risk factors")
    low_risk_factors: List[str] = Field(..., description="Low-priority risk factors")
    
    # Mitigation strategies
    risk_mitigation_strategies: List[str] = Field(..., description="Recommended risk mitigation strategies")
    monitoring_recommendations: List[str] = Field(..., description="Recommended monitoring actions")
    
    # Scenario analysis
    best_case_scenario: str = Field(..., description="Best case outcome description")
    worst_case_scenario: str = Field(..., description="Worst case outcome description")
    most_likely_scenario: str = Field(..., description="Most likely outcome description")
    
    # Dynamic factors
    risk_trend: str = Field(..., description="Risk trend (increasing/stable/decreasing)")
    key_risk_drivers: List[str] = Field(..., description="Primary drivers of current risk level")


class PredictionEngine:
    """
    Advanced prediction engine using Gemini AI and pattern recognition
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        pattern_engine: Optional[PatternRecognitionEngine] = None,
        confidence_threshold: float = 0.6,
        enable_ensemble_predictions: bool = True,
        max_prediction_horizon_days: int = 30
    ):
        """
        Initialize prediction engine
        
        Args:
            gemini_client: Configured Gemini client
            pattern_engine: Pattern recognition engine for context
            confidence_threshold: Minimum confidence for predictions
            enable_ensemble_predictions: Use ensemble methods
            max_prediction_horizon_days: Maximum prediction horizon
        """
        self.gemini_client = gemini_client
        self.pattern_engine = pattern_engine
        self.confidence_threshold = confidence_threshold
        self.enable_ensemble_predictions = enable_ensemble_predictions
        self.max_prediction_horizon_days = max_prediction_horizon_days
        
        # Prediction history for model improvement
        self.prediction_history: Dict[PredictionType, List[Any]] = {
            pred_type: [] for pred_type in PredictionType
        }
        
        # Model performance tracking
        self.model_accuracy: Dict[PredictionType, List[float]] = {
            pred_type: [] for pred_type in PredictionType
        }
        
        # Register prediction functions
        self._register_prediction_functions()
        
        logger.info(
            "Prediction engine initialized",
            confidence_threshold=confidence_threshold,
            ensemble_enabled=enable_ensemble_predictions,
            max_horizon_days=max_prediction_horizon_days
        )
    
    def _register_prediction_functions(self) -> None:
        """Register prediction functions for function calling"""
        
        # Statistical modeling function
        stats_function = FunctionSpec(
            name="calculate_statistical_model",
            description="Calculate statistical model for predictions",
            parameters={
                "type": "object",
                "properties": {
                    "historical_data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Historical data points for modeling"
                    },
                    "model_type": {
                        "type": "string",
                        "enum": ["linear_regression", "arima", "exponential_smoothing", "polynomial"],
                        "description": "Type of statistical model to use"
                    },
                    "prediction_periods": {
                        "type": "integer",
                        "description": "Number of periods to predict forward"
                    }
                },
                "required": ["historical_data", "model_type", "prediction_periods"]
            }
        )
        
        # Market sentiment analysis function
        sentiment_function = FunctionSpec(
            name="analyze_market_sentiment",
            description="Analyze market sentiment for prediction context",
            parameters={
                "type": "object",
                "properties": {
                    "market_indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of market indicators to analyze"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Analysis timeframe"
                    }
                },
                "required": ["market_indicators", "timeframe"]
            }
        )
        
        # Risk quantification function
        risk_function = FunctionSpec(
            name="quantify_risk_metrics",
            description="Quantify specific risk metrics",
            parameters={
                "type": "object",
                "properties": {
                    "risk_factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of risk factors to quantify"
                    },
                    "historical_volatility": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Historical volatility data"
                    },
                    "correlation_matrix": {
                        "type": "object",
                        "description": "Correlation matrix for risk assessment"
                    }
                },
                "required": ["risk_factors"]
            }
        )
        
        self.gemini_client.register_multiple_functions([
            stats_function,
            sentiment_function,
            risk_function
        ])
    
    async def predict_pool_performance(
        self,
        pool_data_history: List[PoolData],
        horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM,
        include_market_context: bool = True
    ) -> PoolPerformancePrediction:
        """
        Predict pool performance metrics
        
        Args:
            pool_data_history: Historical pool data
            horizon: Prediction time horizon
            include_market_context: Include broader market context
            
        Returns:
            Pool performance prediction
        """
        logger.info(
            "Starting pool performance prediction",
            data_points=len(pool_data_history),
            horizon=horizon.value,
            market_context=include_market_context
        )
        
        if len(pool_data_history) < 10:
            raise ValueError("Insufficient historical data for prediction")
        
        # Get the most recent pool for identification
        latest_pool = pool_data_history[-1]
        
        # Prepare historical data for analysis
        historical_summary = self._prepare_pool_prediction_data(pool_data_history)
        
        # Get pattern context if available
        pattern_context = ""
        if self.pattern_engine:
            try:
                patterns = await self.pattern_engine.detect_trading_patterns(
                    pool_data_history[-50:],  # Recent patterns
                    min_volume_threshold=1000.0
                )
                pattern_context = f"\nRecent Patterns Detected: {len(patterns)} trading patterns identified"
                if patterns:
                    pattern_context += f"\nTop Pattern: {patterns[0].pattern_name} (confidence: {patterns[0].confidence:.2f})"
            except Exception as e:
                logger.warning("Failed to get pattern context", error=str(e))
        
        # Create prediction prompt
        prompt = f"""
        Predict the performance of the following DeFi liquidity pool over the next {horizon.value}:
        
        Pool Information:
        - Address: {latest_pool.pool_address}
        - Token Pair: {latest_pool.token0_symbol}/{latest_pool.token1_symbol}
        - Current Price: {latest_pool.price}
        - Current Liquidity: ${latest_pool.liquidity:,.2f}
        - Current 24h Volume: ${latest_pool.volume_24h:,.2f}
        
        Historical Performance Data:
        {json.dumps(historical_summary, indent=2)}
        
        {pattern_context}
        
        Prediction Requirements:
        - Analyze volume trends and seasonal patterns
        - Consider price volatility and market conditions
        - Assess liquidity provider behavior and flows
        - Factor in fee generation potential
        - Evaluate impermanent loss risks
        - Consider broader DeFi market dynamics
        
        Provide quantitative predictions with confidence intervals.
        Include key assumptions and risk factors in your analysis.
        """
        
        try:
            # Generate structured prediction
            prediction = await self.gemini_client.generate_structured_content(
                prompt=prompt,
                output_schema=PoolPerformancePrediction,
                config=ModelConfig(temperature=0.3, max_output_tokens=4096),
                cache_key="pool_prediction_context",
                functions=["calculate_statistical_model", "analyze_market_sentiment"]
            )
            
            # Validate prediction confidence
            if prediction.confidence < self.confidence_threshold:
                logger.warning(
                    "Low confidence pool prediction",
                    confidence=prediction.confidence,
                    threshold=self.confidence_threshold
                )
            
            # Store prediction in history
            self.prediction_history[PredictionType.POOL_PERFORMANCE].append(prediction)
            
            logger.info(
                "Pool performance prediction completed",
                pool_address=prediction.pool_address,
                confidence=prediction.confidence,
                predicted_volume_change=prediction.volume_change_pct
            )
            
            return prediction
            
        except Exception as e:
            logger.error("Pool performance prediction failed", error=str(e))
            raise
    
    async def predict_voting_outcome(
        self,
        proposal_info: Dict[str, Any],
        voting_history: List[VotingData],
        current_votes: Optional[List[VotingData]] = None
    ) -> VotingOutcomePrediction:
        """
        Predict governance voting outcome
        
        Args:
            proposal_info: Information about the proposal
            voting_history: Historical voting data
            current_votes: Current votes for this proposal (if any)
            
        Returns:
            Voting outcome prediction
        """
        logger.info(
            "Starting voting outcome prediction",
            proposal_id=proposal_info.get("proposal_id", "unknown"),
            historical_votes=len(voting_history),
            current_votes=len(current_votes) if current_votes else 0
        )
        
        # Prepare voting analysis data
        voting_summary = self._prepare_voting_prediction_data(voting_history, current_votes)
        
        # Get coalition context if available
        coalition_context = ""
        if self.pattern_engine:
            try:
                coalitions = await self.pattern_engine.detect_voting_coalitions(
                    voting_history,
                    min_proposals=2,
                    min_alignment=0.7
                )
                coalition_context = f"\nDetected Coalitions: {len(coalitions)} voting coalitions identified"
                if coalitions:
                    total_coalition_power = sum(c.total_voting_power for c in coalitions)
                    coalition_context += f"\nTotal Coalition Power: {total_coalition_power:,.0f}"
            except Exception as e:
                logger.warning("Failed to get coalition context", error=str(e))
        
        # Create prediction prompt
        prompt = f"""
        Predict the outcome of the following governance proposal:
        
        Proposal Information:
        {json.dumps(proposal_info, indent=2)}
        
        Historical Voting Analysis:
        {json.dumps(voting_summary, indent=2)}
        
        {coalition_context}
        
        Prediction Requirements:
        - Analyze historical voting patterns and participation rates
        - Consider proposal type and complexity
        - Assess stakeholder alignment and incentives
        - Factor in voting power distribution
        - Consider coalition influence and coordination
        - Evaluate community sentiment and engagement
        - Account for time-sensitive factors and deadline effects
        
        Provide quantitative vote distribution predictions with confidence levels.
        Identify key voters and coalitions that could influence the outcome.
        """
        
        try:
            # Generate structured prediction
            prediction = await self.gemini_client.generate_structured_content(
                prompt=prompt,
                output_schema=VotingOutcomePrediction,
                config=ModelConfig(temperature=0.2, max_output_tokens=4096),
                cache_key="voting_prediction_context",
                functions=["analyze_market_sentiment"]
            )
            
            # Validate prediction confidence
            if prediction.outcome_confidence < self.confidence_threshold:
                logger.warning(
                    "Low confidence voting prediction",
                    confidence=prediction.outcome_confidence,
                    threshold=self.confidence_threshold
                )
            
            # Store prediction in history
            self.prediction_history[PredictionType.VOTING_OUTCOME].append(prediction)
            
            logger.info(
                "Voting outcome prediction completed",
                proposal_id=prediction.proposal_id,
                predicted_outcome=prediction.predicted_outcome,
                confidence=prediction.outcome_confidence
            )
            
            return prediction
            
        except Exception as e:
            logger.error("Voting outcome prediction failed", error=str(e))
            raise
    
    async def analyze_price_impact(
        self,
        pool_data: PoolData,
        trade_size_usd: float,
        trade_direction: str = "buy",
        include_mev_analysis: bool = True
    ) -> PriceImpactAnalysis:
        """
        Analyze price impact of a potential trade
        
        Args:
            pool_data: Current pool state
            trade_size_usd: Trade size in USD
            trade_direction: Trade direction ("buy" or "sell")
            include_mev_analysis: Include MEV risk analysis
            
        Returns:
            Price impact analysis
        """
        logger.info(
            "Starting price impact analysis",
            pool_address=pool_data.pool_address,
            trade_size=trade_size_usd,
            direction=trade_direction
        )
        
        # Calculate basic market metrics
        liquidity_ratio = trade_size_usd / pool_data.liquidity if pool_data.liquidity > 0 else float('inf')
        volume_ratio = trade_size_usd / pool_data.volume_24h if pool_data.volume_24h > 0 else float('inf')
        
        # Create analysis prompt
        prompt = f"""
        Analyze the price impact of a ${trade_size_usd:,.2f} {trade_direction} trade in the following pool:
        
        Pool Details:
        - Address: {pool_data.pool_address}
        - Token Pair: {pool_data.token0_symbol}/{pool_data.token1_symbol}
        - Current Price: {pool_data.price}
        - Total Liquidity: ${pool_data.liquidity:,.2f}
        - 24h Volume: ${pool_data.volume_24h:,.2f}
        - Reserves: {pool_data.reserves[0]:,.2f} {pool_data.token0_symbol}, {pool_data.reserves[1]:,.2f} {pool_data.token1_symbol}
        
        Trade Analysis:
        - Trade Size: ${trade_size_usd:,.2f}
        - Direction: {trade_direction}
        - Liquidity Utilization: {liquidity_ratio:.2%}
        - Volume Ratio: {volume_ratio:.2%}
        
        Analysis Requirements:
        - Calculate precise slippage using constant product formula
        - Estimate price impact and market depth
        - Assess liquidity availability in trade direction
        - Consider recovery time and mechanism
        - Evaluate optimal execution strategies
        {"- Analyze MEV risks (front-running, sandwich attacks)" if include_mev_analysis else ""}
        - Suggest alternative routes or execution methods
        
        Provide quantitative impact estimates and risk assessments.
        """
        
        try:
            # Generate structured analysis
            analysis = await self.gemini_client.generate_structured_content(
                prompt=prompt,
                output_schema=PriceImpactAnalysis,
                config=ModelConfig(temperature=0.1, max_output_tokens=4096),
                functions=["calculate_statistical_model", "quantify_risk_metrics"]
            )
            
            # Store analysis in history
            self.prediction_history[PredictionType.PRICE_IMPACT].append(analysis)
            
            logger.info(
                "Price impact analysis completed",
                pool_address=analysis.pool_address,
                predicted_impact=analysis.price_impact_pct,
                mev_risk=analysis.mev_risk_score
            )
            
            return analysis
            
        except Exception as e:
            logger.error("Price impact analysis failed", error=str(e))
            raise
    
    async def assess_risk(
        self,
        assessment_type: str,
        subject_data: Dict[str, Any],
        timeframe: TimeHorizon = TimeHorizon.MEDIUM_TERM,
        include_scenario_analysis: bool = True
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment
        
        Args:
            assessment_type: Type of risk assessment
            subject_data: Data about the subject being assessed
            timeframe: Assessment time horizon
            include_scenario_analysis: Include scenario analysis
            
        Returns:
            Comprehensive risk assessment
        """
        logger.info(
            "Starting risk assessment",
            type=assessment_type,
            timeframe=timeframe.value,
            scenario_analysis=include_scenario_analysis
        )
        
        # Create comprehensive risk analysis prompt
        prompt = f"""
        Perform a comprehensive risk assessment for the following:
        
        Assessment Type: {assessment_type}
        Time Horizon: {timeframe.value}
        
        Subject Data:
        {json.dumps(subject_data, indent=2)}
        
        Risk Assessment Requirements:
        - Evaluate market risk (price volatility, correlation risks)
        - Assess liquidity risk (withdrawal risks, market depth)
        - Analyze operational risk (smart contract bugs, admin keys)
        - Consider regulatory risk (compliance, regulatory changes)  
        - Evaluate counterparty risk (protocol risk, oracle risk)
        - Quantify overall risk score and categorization
        
        {"Scenario Analysis Requirements:" if include_scenario_analysis else ""}
        {"- Model best-case, worst-case, and most likely scenarios" if include_scenario_analysis else ""}
        {"- Consider tail risks and black swan events" if include_scenario_analysis else ""}
        {"- Analyze correlation breakdowns during stress periods" if include_scenario_analysis else ""}
        
        Provide specific risk mitigation strategies and monitoring recommendations.
        Quantify risk scores on a 0-1 scale with clear explanations.
        """
        
        try:
            # Generate structured assessment
            assessment = await self.gemini_client.generate_structured_content(
                prompt=prompt,
                output_schema=RiskAssessment,
                config=ModelConfig(temperature=0.2, max_output_tokens=4096),
                functions=["quantify_risk_metrics", "analyze_market_sentiment"]
            )
            
            # Store assessment in history
            self.prediction_history[PredictionType.RISK_ASSESSMENT].append(assessment)
            
            logger.info(
                "Risk assessment completed",
                type=assessment_type,
                risk_level=assessment.risk_level,
                overall_score=assessment.overall_risk_score
            )
            
            return assessment
            
        except Exception as e:
            logger.error("Risk assessment failed", error=str(e))
            raise
    
    def _prepare_pool_prediction_data(self, pool_data: List[PoolData]) -> Dict[str, Any]:
        """Prepare pool data for prediction analysis"""
        if not pool_data:
            return {}
        
        # Convert to DataFrame for statistical analysis
        df_data = [pool.to_dict() for pool in pool_data]
        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate technical indicators
        df['volume_ma_7'] = df['volume_24h'].rolling(window=7, min_periods=1).mean()
        df['price_volatility'] = df['price'].rolling(window=7, min_periods=1).std() / df['price'].rolling(window=7, min_periods=1).mean()
        df['liquidity_trend'] = df['liquidity'].pct_change().rolling(window=7, min_periods=1).mean()
        
        return {
            "data_points": len(df),
            "time_span_days": (df['timestamp'].max() - df['timestamp'].min()).days,
            "current_metrics": {
                "price": float(df['price'].iloc[-1]),
                "volume_24h": float(df['volume_24h'].iloc[-1]),
                "liquidity": float(df['liquidity'].iloc[-1]),
                "fees_24h": float(df['fees_24h'].iloc[-1])
            },
            "statistical_summary": {
                "price_mean": float(df['price'].mean()),
                "price_std": float(df['price'].std()),
                "volume_mean": float(df['volume_24h'].mean()),
                "volume_std": float(df['volume_24h'].std()),
                "liquidity_mean": float(df['liquidity'].mean()),
                "current_price_zscore": float((df['price'].iloc[-1] - df['price'].mean()) / df['price'].std()) if df['price'].std() > 0 else 0
            },
            "trend_analysis": {
                "price_trend_7d": float(df['price'].pct_change().rolling(7).mean().iloc[-1]) if len(df) >= 7 else 0,
                "volume_trend_7d": float(df['volume_24h'].pct_change().rolling(7).mean().iloc[-1]) if len(df) >= 7 else 0,
                "liquidity_trend_7d": float(df['liquidity_trend'].iloc[-1]) if not pd.isna(df['liquidity_trend'].iloc[-1]) else 0,
                "volatility_trend": float(df['price_volatility'].iloc[-1]) if not pd.isna(df['price_volatility'].iloc[-1]) else 0
            },
            "recent_performance": {
                "price_change_24h": float(df['price'].pct_change().iloc[-1]) if len(df) > 1 else 0,
                "volume_change_24h": float(df['volume_24h'].pct_change().iloc[-1]) if len(df) > 1 else 0,
                "liquidity_change_24h": float(df['liquidity'].pct_change().iloc[-1]) if len(df) > 1 else 0
            }
        }
    
    def _prepare_voting_prediction_data(
        self, 
        voting_history: List[VotingData], 
        current_votes: Optional[List[VotingData]] = None
    ) -> Dict[str, Any]:
        """Prepare voting data for prediction analysis"""
        if not voting_history:
            return {}
        
        # Convert to DataFrame for analysis
        df_data = [vote.to_dict() for vote in voting_history]
        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Current vote analysis
        current_vote_summary = {}
        if current_votes:
            current_df = pd.DataFrame([vote.to_dict() for vote in current_votes])
            current_vote_summary = {
                "votes_cast": len(current_df),
                "total_voting_power": float(current_df['voting_power'].sum()),
                "vote_distribution": current_df['vote_choice'].value_counts().to_dict(),
                "power_distribution": current_df.groupby('vote_choice')['voting_power'].sum().to_dict()
            }
        
        return {
            "historical_summary": {
                "total_votes": len(df),
                "unique_voters": df['voter_address'].nunique(),
                "unique_proposals": df['proposal_id'].nunique(),
                "avg_participation_rate": float(df.groupby('proposal_id')['voter_address'].count().mean()),
                "avg_voting_power_per_vote": float(df['voting_power'].mean()),
                "historical_outcomes": df.groupby('proposal_id')['vote_choice'].apply(lambda x: x.value_counts().to_dict()).to_dict()
            },
            "voter_behavior": {
                "top_voters_by_power": df.groupby('voter_address')['voting_power'].sum().nlargest(20).to_dict(),
                "most_active_voters": df['voter_address'].value_counts().head(20).to_dict(),
                "vote_choice_distribution": df['vote_choice'].value_counts(normalize=True).to_dict()
            },
            "temporal_patterns": {
                "votes_by_day": df.groupby(df['timestamp'].dt.date)['voter_address'].count().to_dict(),
                "voting_power_by_day": df.groupby(df['timestamp'].dt.date)['voting_power'].sum().to_dict()
            },
            "current_state": current_vote_summary
        }
    
    async def get_prediction_accuracy(self, prediction_type: PredictionType) -> Dict[str, float]:
        """Get accuracy metrics for a prediction type"""
        accuracies = self.model_accuracy.get(prediction_type, [])
        
        if not accuracies:
            return {"accuracy": 0.0, "count": 0}
        
        return {
            "accuracy": np.mean(accuracies),
            "std_dev": np.std(accuracies),
            "count": len(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies)
        }
    
    async def update_model_accuracy(
        self, 
        prediction_type: PredictionType, 
        actual_outcome: Any, 
        prediction: Any
    ) -> float:
        """Update model accuracy based on actual outcomes"""
        try:
            # Calculate accuracy based on prediction type
            if prediction_type == PredictionType.POOL_PERFORMANCE:
                # For pool performance, calculate accuracy of volume and price predictions
                volume_accuracy = 1.0 - abs(actual_outcome.volume_24h - prediction.predicted_volume_24h) / actual_outcome.volume_24h
                price_accuracy = 1.0 - abs(actual_outcome.price - prediction.predicted_price) / actual_outcome.price
                accuracy = (volume_accuracy + price_accuracy) / 2.0
                
            elif prediction_type == PredictionType.VOTING_OUTCOME:
                # For voting, binary accuracy on outcome
                accuracy = 1.0 if actual_outcome.outcome == prediction.predicted_outcome else 0.0
                
            elif prediction_type == PredictionType.PRICE_IMPACT:
                # For price impact, accuracy of impact percentage
                accuracy = 1.0 - abs(actual_outcome.actual_impact_pct - prediction.price_impact_pct) / 100.0
                
            else:
                # Generic accuracy calculation
                accuracy = 0.8  # Placeholder
            
            # Clamp accuracy between 0 and 1
            accuracy = max(0.0, min(1.0, accuracy))
            
            # Store accuracy
            self.model_accuracy[prediction_type].append(accuracy)
            
            # Keep only recent accuracy measurements (last 100)
            if len(self.model_accuracy[prediction_type]) > 100:
                self.model_accuracy[prediction_type] = self.model_accuracy[prediction_type][-100:]
            
            logger.info(
                "Model accuracy updated",
                prediction_type=prediction_type.value,
                accuracy=accuracy,
                total_measurements=len(self.model_accuracy[prediction_type])
            )
            
            return accuracy
            
        except Exception as e:
            logger.error("Failed to update model accuracy", error=str(e))
            return 0.0
    
    async def export_predictions_to_cache(self, cache_key: str) -> bool:
        """Export predictions to Gemini cache for future analysis"""
        try:
            prediction_export = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "predictions": {},
                "accuracy_metrics": {}
            }
            
            # Export recent predictions
            for pred_type in PredictionType:
                recent_predictions = self.prediction_history[pred_type][-50:]  # Last 50 predictions
                prediction_export["predictions"][pred_type.value] = [
                    pred.dict() if hasattr(pred, 'dict') else str(pred)
                    for pred in recent_predictions
                ]
                
                # Export accuracy metrics
                accuracy_metrics = await self.get_prediction_accuracy(pred_type)
                prediction_export["accuracy_metrics"][pred_type.value] = accuracy_metrics
            
            export_content = [json.dumps(prediction_export, indent=2)]
            
            cache_name = await self.gemini_client.create_cache(
                cache_key=cache_key,
                content=export_content,
                ttl_hours=48
            )
            
            if cache_name:
                logger.info("Predictions exported to cache", cache_key=cache_key, cache_name=cache_name)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error("Failed to export predictions to cache", error=str(e))
            return False
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get prediction engine statistics"""
        return {
            "total_predictions": sum(len(preds) for preds in self.prediction_history.values()),
            "predictions_by_type": {
                pred_type.value: len(self.prediction_history[pred_type])
                for pred_type in PredictionType
            },
            "accuracy_by_type": {
                pred_type.value: self.model_accuracy[pred_type][-1] if self.model_accuracy[pred_type] else 0.0
                for pred_type in PredictionType
            },
            "confidence_threshold": self.confidence_threshold,
            "ensemble_enabled": self.enable_ensemble_predictions,
            "max_horizon_days": self.max_prediction_horizon_days
        }