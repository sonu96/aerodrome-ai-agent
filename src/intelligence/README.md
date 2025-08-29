# Aerodrome AI Intelligence System

A sophisticated AI-powered intelligence system built on Google Gemini 2.0 models, designed for advanced pattern recognition, predictive analytics, and comprehensive market analysis in the Aerodrome Finance ecosystem.

## 🚀 Features

### Advanced Gemini 2.0 Integration
- **Latest Models**: Support for Gemini 2.0 Flash and Pro models with cutting-edge capabilities
- **Function Calling**: Automatic function execution for complex calculations and data analysis
- **Context Caching**: Cost-effective analysis with intelligent content caching (up to 48-hour TTL)
- **Streaming Responses**: Real-time response streaming for improved user experience
- **Structured Output**: Type-safe structured responses using Pydantic models
- **Safety Controls**: Configurable safety settings for different use cases

### Pattern Recognition Engine
- **Trading Pattern Detection**: Identify complex trading patterns, MEV activities, and market manipulations
- **Voting Coalition Analysis**: Detect governance coalitions and voting behavior patterns
- **Arbitrage Opportunities**: Real-time arbitrage detection across multiple pools
- **Market Correlations**: Advanced correlation analysis between assets and markets
- **Anomaly Detection**: Identify unusual activities and potential risks

### Prediction Engine
- **Pool Performance Prediction**: Forecast volume, liquidity, and fee generation
- **Voting Outcome Prediction**: Predict governance proposal outcomes with confidence intervals
- **Price Impact Analysis**: Detailed analysis of trade impact and slippage estimation
- **Risk Assessment**: Comprehensive multi-dimensional risk evaluation
- **Scenario Analysis**: Best/worst/most likely outcome modeling

## 📋 Requirements

```
python-dotenv>=1.0.0
pydantic>=2.5.0
google-generativeai>=0.8.3
pandas>=2.1.4
numpy>=1.26.2
structlog>=23.2.0
aiohttp>=3.9.1
```

## 🛠 Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Google AI API key:
```bash
export GOOGLE_AI_API_KEY="your_api_key_here"
```

3. Import the modules:
```python
from src.intelligence import GeminiClient, PatternRecognitionEngine, PredictionEngine
```

## 🔧 Quick Start

### Basic Gemini Client Usage

```python
import asyncio
from src.intelligence import GeminiClient, GeminiModel, SafetyLevel

async def basic_example():
    # Initialize client
    client = GeminiClient(
        api_key="your_api_key",
        default_model=GeminiModel.FLASH,
        safety_level=SafetyLevel.DEFAULT,
        enable_caching=True
    )
    
    # Generate content
    response = await client.generate_content(
        prompt="Explain impermanent loss in DeFi",
        config=ModelConfig(temperature=0.3)
    )
    
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Cached: {response.cached}")

asyncio.run(basic_example())
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel, Field

class MarketAnalysis(BaseModel):
    sentiment: str = Field(description="Market sentiment")
    risk_level: str = Field(description="Risk assessment")
    confidence: float = Field(ge=0.0, le=1.0)

async def structured_example():
    client = GeminiClient(api_key="your_api_key")
    
    analysis = await client.generate_structured_content(
        prompt="Analyze the current DeFi market",
        output_schema=MarketAnalysis
    )
    
    print(f"Sentiment: {analysis.sentiment}")
    print(f"Risk: {analysis.risk_level}")
```

### Pattern Recognition

```python
from src.intelligence import PatternRecognitionEngine, PoolData
from datetime import datetime

async def pattern_example():
    # Initialize engines
    gemini_client = GeminiClient(api_key="your_api_key")
    pattern_engine = PatternRecognitionEngine(
        gemini_client=gemini_client,
        min_confidence=0.7
    )
    
    # Create sample pool data
    pool_data = [
        PoolData(
            pool_address="0x...",
            token0_symbol="ETH",
            token1_symbol="USDC",
            reserves=(1000.0, 1800000.0),
            price=1800.0,
            volume_24h=50000.0,
            fees_24h=150.0,
            liquidity=2000000.0,
            timestamp=datetime.utcnow()
        )
        # Add more historical data...
    ]
    
    # Detect trading patterns
    patterns = await pattern_engine.detect_trading_patterns(
        pool_data_history=pool_data,
        min_volume_threshold=10000.0
    )
    
    for pattern in patterns:
        print(f"Pattern: {pattern.pattern_name}")
        print(f"Confidence: {pattern.confidence:.2f}")
        print(f"Impact: ${pattern.volume_impact:,.2f}")
```

### Predictions

```python
from src.intelligence import PredictionEngine, TimeHorizon

async def prediction_example():
    # Initialize prediction engine
    prediction_engine = PredictionEngine(
        gemini_client=gemini_client,
        confidence_threshold=0.6
    )
    
    # Predict pool performance
    prediction = await prediction_engine.predict_pool_performance(
        pool_data_history=historical_data,
        horizon=TimeHorizon.MEDIUM_TERM
    )
    
    print(f"Predicted volume change: {prediction.volume_change_pct:.1f}%")
    print(f"Predicted APR: {prediction.apr_estimate:.2f}%")
    print(f"Confidence: {prediction.confidence:.2f}")
    
    # Analyze price impact
    impact = await prediction_engine.analyze_price_impact(
        pool_data=current_pool_state,
        trade_size_usd=50000.0,
        trade_direction="buy"
    )
    
    print(f"Price impact: {impact.price_impact_pct:.2f}%")
    print(f"MEV risk: {impact.mev_risk_score:.2f}")
```

## 🔍 Advanced Features

### Context Caching

Reduce costs and improve performance with intelligent caching:

```python
# Create cache
cache_name = await client.create_cache(
    cache_key="market_data",
    content=["Market data content..."],
    ttl_hours=24
)

# Use cached content
response = await client.generate_content(
    prompt="Analyze this market data",
    cache_key="market_data"
)
```

### Function Calling

Register custom functions for complex calculations:

```python
from src.intelligence import FunctionSpec

# Define function
calc_function = FunctionSpec(
    name="calculate_apy",
    description="Calculate APY from returns",
    parameters={
        "type": "object",
        "properties": {
            "daily_returns": {
                "type": "array",
                "items": {"type": "number"}
            }
        }
    }
)

client.register_function(calc_function)

# Use in generation
response = await client.generate_content(
    prompt="Calculate APY for daily returns: [0.01, 0.02, 0.015]",
    functions=["calculate_apy"]
)
```

### Streaming Responses

Get real-time responses for better UX:

```python
async for chunk in client.stream_content(
    prompt="Provide detailed market analysis"
):
    print(chunk, end="", flush=True)
```

## 📊 Model Performance Tracking

The prediction engine includes built-in accuracy tracking:

```python
# Get accuracy metrics
accuracy = await prediction_engine.get_prediction_accuracy(
    PredictionType.POOL_PERFORMANCE
)

print(f"Average accuracy: {accuracy['accuracy']:.2%}")
print(f"Prediction count: {accuracy['count']}")

# Update accuracy with actual outcomes
await prediction_engine.update_model_accuracy(
    prediction_type=PredictionType.POOL_PERFORMANCE,
    actual_outcome=actual_data,
    prediction=previous_prediction
)
```

## 🛡 Safety and Risk Management

### Configurable Safety Levels

```python
# Strict safety (blocks low-risk content)
client = GeminiClient(
    api_key="key",
    safety_level=SafetyLevel.STRICT
)

# Permissive safety (minimal blocking)
client = GeminiClient(
    api_key="key", 
    safety_level=SafetyLevel.PERMISSIVE
)
```

### Rate Limiting

Built-in rate limiting prevents API abuse:
- Minimum 100ms between requests
- Configurable request timeouts
- Automatic retry logic

### Error Handling

Comprehensive error handling and logging:

```python
try:
    result = await client.generate_content(prompt="...")
except Exception as e:
    logger.error("Generation failed", error=str(e))
    # Fallback logic
```

## 📈 Use Cases

### 1. Trading Strategy Development
- Identify profitable trading patterns
- Analyze market correlations
- Assess risk/reward ratios
- Optimize execution strategies

### 2. Liquidity Pool Analysis
- Predict pool performance
- Calculate impermanent loss risks
- Optimize fee tier selection
- Monitor liquidity flows

### 3. Governance Analysis
- Predict voting outcomes
- Identify influential coalitions
- Track proposal success rates
- Analyze voter behavior

### 4. Risk Management
- Multi-dimensional risk assessment
- Scenario analysis and stress testing
- Real-time risk monitoring
- Mitigation strategy recommendations

## 🔧 Configuration Options

### GeminiClient Configuration

```python
client = GeminiClient(
    api_key="your_key",
    default_model=GeminiModel.FLASH,  # or GeminiModel.PRO
    safety_level=SafetyLevel.DEFAULT,
    enable_caching=True,
    cache_ttl_hours=24,
    request_timeout=60
)
```

### PatternRecognitionEngine Configuration

```python
pattern_engine = PatternRecognitionEngine(
    gemini_client=client,
    analysis_window_hours=24,
    min_confidence=0.7,
    enable_advanced_analytics=True
)
```

### PredictionEngine Configuration

```python
prediction_engine = PredictionEngine(
    gemini_client=client,
    pattern_engine=pattern_engine,
    confidence_threshold=0.6,
    enable_ensemble_predictions=True,
    max_prediction_horizon_days=30
)
```

## 🔍 Monitoring and Observability

### Built-in Logging

Structured logging with contextual information:

```python
import structlog

logger = structlog.get_logger(__name__)
# Logs include request IDs, timing, and metadata
```

### Performance Metrics

Track key performance indicators:

```python
# Client statistics
stats = client.get_stats()
print(f"Model: {stats['default_model']}")
print(f"Cache usage: {stats['cached_content_count']}")

# Engine statistics  
engine_stats = prediction_engine.get_engine_stats()
print(f"Total predictions: {engine_stats['total_predictions']}")
print(f"Accuracy by type: {engine_stats['accuracy_by_type']}")
```

## 🚨 Best Practices

### 1. Cost Optimization
- Use caching for repeated analysis
- Choose appropriate models for tasks
- Implement request batching where possible

### 2. Accuracy Improvement
- Maintain sufficient historical data
- Regular model accuracy evaluation
- Ensemble predictions for critical decisions

### 3. Error Resilience
- Implement proper exception handling
- Use fallback strategies
- Monitor API rate limits

### 4. Security
- Secure API key management
- Input validation and sanitization
- Appropriate safety level configuration

## 📚 API Reference

### Core Classes

- **GeminiClient**: Advanced Gemini 2.0 client with enterprise features
- **PatternRecognitionEngine**: AI-powered pattern detection system
- **PredictionEngine**: Predictive analytics and forecasting engine

### Data Models

- **PoolData**: Liquidity pool state and metrics
- **VotingData**: Governance voting records
- **TradingPattern**: Detected trading pattern information
- **VotingCoalition**: Coalition analysis results
- **ArbitrageOpportunity**: Arbitrage opportunity details
- **PoolPerformancePrediction**: Pool performance forecast
- **VotingOutcomePrediction**: Voting outcome forecast
- **PriceImpactAnalysis**: Trade impact analysis
- **RiskAssessment**: Comprehensive risk evaluation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is part of the Aerodrome AI Agent system. See the main project license for details.

## 🔗 Related Documentation

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Aerodrome Finance Documentation](https://docs.aerodrome.finance/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

For more examples and advanced usage patterns, see the `example_usage.py` file in this directory.