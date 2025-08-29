"""
Example Usage of Aerodrome AI Intelligence System

This example demonstrates how to use the sophisticated Gemini 2.0 integration
for pattern recognition, prediction, and analysis.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import List

from gemini_client import GeminiClient, GeminiModel, ModelConfig, SafetyLevel
from pattern_recognition import PatternRecognitionEngine, PoolData, VotingData
from prediction_engine import PredictionEngine, TimeHorizon


async def example_gemini_client_usage():
    """Example: Advanced Gemini client usage"""
    print("=== Gemini Client Example ===")
    
    # Initialize client with advanced settings
    client = GeminiClient(
        api_key=os.getenv("GOOGLE_AI_API_KEY"),
        default_model=GeminiModel.FLASH,
        safety_level=SafetyLevel.DEFAULT,
        enable_caching=True,
        cache_ttl_hours=24
    )
    
    # Basic content generation
    response = await client.generate_content(
        prompt="Explain the concept of impermanent loss in DeFi liquidity pools",
        config=ModelConfig(temperature=0.3)
    )
    print(f"Response: {response.content[:200]}...")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Response time: {response.response_time:.2f}s")
    
    # Structured output example
    from pydantic import BaseModel, Field
    
    class MarketSummary(BaseModel):
        market_sentiment: str = Field(description="Overall market sentiment")
        key_trends: List[str] = Field(description="Key market trends")
        risk_level: str = Field(description="Current risk level")
        confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence")
    
    structured_response = await client.generate_structured_content(
        prompt="Analyze the current DeFi market conditions and provide a structured summary",
        output_schema=MarketSummary
    )
    print(f"Market Sentiment: {structured_response.market_sentiment}")
    print(f"Risk Level: {structured_response.risk_level}")
    
    # Streaming example
    print("\n=== Streaming Response ===")
    async for chunk in client.stream_content(
        prompt="Provide a detailed explanation of Aerodrome Finance's ve(3,3) model"
    ):
        print(chunk, end="", flush=True)
    print("\n")


async def example_pattern_recognition():
    """Example: Pattern recognition engine"""
    print("\n=== Pattern Recognition Example ===")
    
    # Initialize clients
    gemini_client = GeminiClient(
        api_key=os.getenv("GOOGLE_AI_API_KEY"),
        default_model=GeminiModel.FLASH
    )
    
    pattern_engine = PatternRecognitionEngine(
        gemini_client=gemini_client,
        analysis_window_hours=24,
        min_confidence=0.7
    )
    
    # Generate sample pool data
    sample_pools = generate_sample_pool_data()
    
    # Detect trading patterns
    trading_patterns = await pattern_engine.detect_trading_patterns(
        pool_data_history=sample_pools,
        min_volume_threshold=10000.0
    )
    
    print(f"Detected {len(trading_patterns)} trading patterns")
    for pattern in trading_patterns:
        print(f"- {pattern.pattern_name}: {pattern.description[:100]}...")
        print(f"  Confidence: {pattern.confidence:.2f}, Strength: {pattern.strength:.2f}")
    
    # Detect arbitrage opportunities
    arbitrage_opportunities = await pattern_engine.detect_arbitrage_opportunities(
        pool_data=sample_pools[-10:],  # Recent pools
        min_profit_threshold=50.0
    )
    
    print(f"\nDetected {len(arbitrage_opportunities)} arbitrage opportunities")
    for opp in arbitrage_opportunities:
        print(f"- {opp.opportunity_type}: ${opp.potential_profit:.2f} profit")
        print(f"  ROI: {opp.roi_percentage:.1f}%, Risk Score: {opp.confidence:.2f}")
    
    # Generate sample voting data
    sample_votes = generate_sample_voting_data()
    
    # Detect voting coalitions
    coalitions = await pattern_engine.detect_voting_coalitions(
        voting_history=sample_votes,
        min_proposals=3,
        min_alignment=0.8
    )
    
    print(f"\nDetected {len(coalitions)} voting coalitions")
    for coalition in coalitions:
        print(f"- {coalition.coalition_name}: {len(coalition.members)} members")
        print(f"  Total Power: {coalition.total_voting_power:,.0f}, Alignment: {coalition.alignment_percentage:.1f}%")


async def example_prediction_engine():
    """Example: Prediction engine usage"""
    print("\n=== Prediction Engine Example ===")
    
    # Initialize clients
    gemini_client = GeminiClient(
        api_key=os.getenv("GOOGLE_AI_API_KEY"),
        default_model=GeminiModel.FLASH
    )
    
    pattern_engine = PatternRecognitionEngine(gemini_client)
    prediction_engine = PredictionEngine(
        gemini_client=gemini_client,
        pattern_engine=pattern_engine,
        confidence_threshold=0.6
    )
    
    # Generate sample data
    sample_pools = generate_sample_pool_data()
    sample_votes = generate_sample_voting_data()
    
    # Pool performance prediction
    pool_prediction = await prediction_engine.predict_pool_performance(
        pool_data_history=sample_pools,
        horizon=TimeHorizon.MEDIUM_TERM
    )
    
    print("Pool Performance Prediction:")
    print(f"- Pool: {pool_prediction.token_pair}")
    print(f"- Predicted Volume Change: {pool_prediction.volume_change_pct:.1f}%")
    print(f"- Predicted Price Change: {pool_prediction.price_change_pct:.1f}%")
    print(f"- Estimated APR: {pool_prediction.apr_estimate:.2f}%")
    print(f"- Confidence: {pool_prediction.confidence:.2f}")
    
    # Price impact analysis
    price_impact = await prediction_engine.analyze_price_impact(
        pool_data=sample_pools[-1],
        trade_size_usd=50000.0,
        trade_direction="buy"
    )
    
    print(f"\nPrice Impact Analysis:")
    print(f"- Trade Size: ${price_impact.trade_size_usd:,.2f}")
    print(f"- Predicted Impact: {price_impact.price_impact_pct:.2f}%")
    print(f"- Slippage Estimate: {price_impact.slippage_estimate:.2f}%")
    print(f"- MEV Risk: {price_impact.mev_risk_score:.2f}")
    print(f"- Recovery Time: {price_impact.price_recovery_time}")
    
    # Voting outcome prediction
    proposal_info = {
        "proposal_id": "PROP-001",
        "title": "Increase rewards for ETH/USDC pool",
        "description": "Proposal to increase weekly rewards allocation",
        "voting_deadline": (datetime.utcnow() + timedelta(days=7)).isoformat()
    }
    
    voting_prediction = await prediction_engine.predict_voting_outcome(
        proposal_info=proposal_info,
        voting_history=sample_votes
    )
    
    print(f"\nVoting Outcome Prediction:")
    print(f"- Proposal: {voting_prediction.proposal_title}")
    print(f"- Predicted Outcome: {voting_prediction.predicted_outcome}")
    print(f"- Confidence: {voting_prediction.outcome_confidence:.2f}")
    print(f"- Expected For/Against: {voting_prediction.predicted_for_pct:.1f}%/{voting_prediction.predicted_against_pct:.1f}%")
    print(f"- Predicted Turnout: {voting_prediction.predicted_turnout:.1f}%")
    
    # Risk assessment
    risk_assessment = await prediction_engine.assess_risk(
        assessment_type="liquidity_pool",
        subject_data={
            "pool_address": sample_pools[-1].pool_address,
            "token_pair": f"{sample_pools[-1].token0_symbol}/{sample_pools[-1].token1_symbol}",
            "current_liquidity": sample_pools[-1].liquidity,
            "volume_24h": sample_pools[-1].volume_24h
        },
        timeframe=TimeHorizon.MEDIUM_TERM
    )
    
    print(f"\nRisk Assessment:")
    print(f"- Overall Risk: {risk_assessment.risk_level}")
    print(f"- Risk Score: {risk_assessment.overall_risk_score:.2f}")
    print(f"- Market Risk: {risk_assessment.market_risk:.2f}")
    print(f"- Liquidity Risk: {risk_assessment.liquidity_risk:.2f}")
    print(f"- Key Risk Factors: {', '.join(risk_assessment.high_risk_factors[:3])}")


def generate_sample_pool_data() -> List[PoolData]:
    """Generate sample pool data for testing"""
    pools = []
    base_time = datetime.utcnow() - timedelta(days=7)
    
    for i in range(50):
        # Simulate price movement with some volatility
        base_price = 1800 + (i * 2) + (i % 10 - 5) * 10
        volume = 50000 + (i * 1000) + (i % 20 - 10) * 5000
        liquidity = 2000000 + (i * 10000) + (i % 15 - 7) * 50000
        
        pool = PoolData(
            pool_address=f"0x{''.join(['a'] * 40)}",
            token0_symbol="ETH",
            token1_symbol="USDC",
            reserves=(liquidity / base_price / 2, liquidity / 2),
            price=base_price,
            volume_24h=max(1000, volume),
            fees_24h=max(10, volume * 0.003),
            liquidity=max(100000, liquidity),
            timestamp=base_time + timedelta(hours=i * 3)
        )
        pools.append(pool)
    
    return pools


def generate_sample_voting_data() -> List[VotingData]:
    """Generate sample voting data for testing"""
    votes = []
    base_time = datetime.utcnow() - timedelta(days=30)
    
    proposals = ["PROP-001", "PROP-002", "PROP-003", "PROP-004", "PROP-005"]
    voters = [f"0x{''.join([hex(i)[2:] for _ in range(20)])}" for i in range(50)]
    choices = ["for", "against", "abstain"]
    
    for i, proposal in enumerate(proposals):
        # Generate votes for each proposal
        for j, voter in enumerate(voters):
            if (i + j) % 3 == 0:  # Not all voters participate in all proposals
                continue
                
            vote = VotingData(
                proposal_id=proposal,
                voter_address=voter,
                voting_power=1000 + (j * 100) + (i * 50),
                vote_choice=choices[(i + j) % 3],
                timestamp=base_time + timedelta(days=i * 5, hours=j),
                block_number=18000000 + (i * 1000) + j
            )
            votes.append(vote)
    
    return votes


async def example_advanced_features():
    """Example: Advanced features and caching"""
    print("\n=== Advanced Features Example ===")
    
    client = GeminiClient(
        api_key=os.getenv("GOOGLE_AI_API_KEY"),
        enable_caching=True
    )
    
    # Create cache for repeated analysis
    cache_key = "market_analysis"
    market_data = [
        "Current market conditions show high volatility in DeFi tokens",
        "Liquidity has increased 15% over the past week",
        "New protocols are launching with innovative tokenomics"
    ]
    
    cache_name = await client.create_cache(
        cache_key=cache_key,
        content=market_data,
        ttl_hours=24
    )
    print(f"Created cache: {cache_name}")
    
    # Use cached content for analysis
    cached_response = await client.generate_content(
        prompt="Based on the market data, what are the key opportunities for yield farming?",
        cache_key=cache_key
    )
    print(f"Cached analysis: {cached_response.content[:200]}...")
    print(f"Used cache: {cached_response.cached}")
    
    # Function calling example
    from gemini_client import FunctionSpec
    
    # Register a custom function
    calculate_apy = FunctionSpec(
        name="calculate_apy",
        description="Calculate APY from daily returns",
        parameters={
            "type": "object",
            "properties": {
                "daily_returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of daily return percentages"
                }
            },
            "required": ["daily_returns"]
        }
    )
    
    client.register_function(calculate_apy)
    
    # Use function in generation
    function_response = await client.generate_content(
        prompt="Calculate the APY for a pool with the following daily returns: [0.05, 0.03, 0.07, 0.02, 0.04]",
        functions=["calculate_apy"]
    )
    
    print(f"Function calls made: {len(function_response.function_calls)}")
    for call in function_response.function_calls:
        print(f"- Called: {call['name']} with args: {call['args']}")
    
    # Get client statistics
    stats = client.get_stats()
    print(f"\nClient Statistics:")
    print(f"- Model: {stats['default_model']}")
    print(f"- Cached content: {stats['cached_content_count']}")
    print(f"- Registered functions: {stats['registered_functions_count']}")


async def main():
    """Run all examples"""
    if not os.getenv("GOOGLE_AI_API_KEY"):
        print("Please set GOOGLE_AI_API_KEY environment variable")
        return
    
    try:
        await example_gemini_client_usage()
        await example_pattern_recognition()
        await example_prediction_engine()
        await example_advanced_features()
        
        print("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())