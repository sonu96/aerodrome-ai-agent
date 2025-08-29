# Aerodrome Protocol Intelligence Brain

An advanced AI-powered knowledge system for the Aerodrome Finance protocol, featuring intelligent memory management with pruning and confidence scoring.

## Overview

This brain is a comprehensive intelligence system that:
- **Knows** everything about the Aerodrome protocol in real-time
- **Learns** from patterns and historical data
- **Remembers** important insights while pruning irrelevant information
- **Provides** confidence-scored analysis and predictions

## Key Features

### 🧠 Complete Protocol Knowledge
- Real-time tracking of all pools, TVL, volumes, and fees
- Understanding of veAERO mechanics and voting patterns
- Emission distribution and gauge weight monitoring
- Historical pattern recognition and trend analysis

### 💾 Intelligent Memory Management
- **Tiered Memory System**: Critical, High, Medium, and Low importance levels
- **Automatic Pruning**: Removes low-confidence and outdated information
- **Memory Consolidation**: Merges similar patterns into higher-confidence insights
- **Adaptive Learning**: Adjusts confidence based on prediction accuracy

### 📊 Confidence Scoring
- Every piece of knowledge has a confidence score (0.0 - 1.0)
- Factors: data reliability, historical accuracy, recency, corroboration
- Self-adjusting based on prediction outcomes
- Enables risk-adjusted decision making

### 🤖 AI-Powered Intelligence
- Google Gemini integration for advanced analysis
- Pattern recognition across complex protocol interactions
- Predictive insights with confidence levels
- Natural language query interface

## Architecture

```
aerodrome-brain/
├── brain/              # Core intelligence system
├── memory/            # Mem0 integration with pruning
├── protocol/          # Aerodrome protocol interface
├── intelligence/      # Gemini AI and analysis
├── api/              # REST/WebSocket APIs
└── config/           # Configuration and rules
```

## Memory Categories

| Category | Confidence Threshold | Retention | Examples |
|----------|---------------------|-----------|----------|
| Protocol Constants | 0.95+ | Permanent | Contract addresses, core functions |
| Pool Performance | 0.70+ | 30 days | Volume patterns, fee trends |
| Voting Patterns | 0.60+ | 14 days | Bribe effectiveness, voter behavior |
| Market Correlations | 0.50+ | 7 days | Price impacts, liquidity flows |
| Speculative Insights | 0.30+ | 2 days | Short-term predictions, anomalies |

## Setup

### Prerequisites
- Python 3.11+
- Google Cloud Project with Gemini API access
- QuickNode endpoint with Aerodrome API addon
- Mem0 API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aerodrome-brain.git
cd aerodrome-brain

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Configuration

```bash
# Required environment variables
QUICKNODE_ENDPOINT=your_quicknode_endpoint
MEM0_API_KEY=your_mem0_key
GOOGLE_CLOUD_PROJECT=your_project_id
GEMINI_API_KEY=your_gemini_key
BASE_RPC_URL=your_base_rpc
```

## Usage

### Starting the Brain

```python
from brain import AerodromeBrain

# Initialize the brain
brain = AerodromeBrain()

# Start monitoring
await brain.start()

# Query the brain
result = await brain.query(
    "What are the most profitable pools based on last week's data?",
    min_confidence=0.7
)
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
```

### API Endpoints

```bash
# Query the brain
GET /api/query?q=<question>&min_confidence=0.7

# Get pool insights
GET /api/pools/<address>/insights

# Get voting recommendations
GET /api/voting/recommendations

# Memory statistics
GET /api/memory/stats
```

## Deployment

### Google Cloud Run

```bash
# Build and deploy
gcloud run deploy aerodrome-brain \
  --source . \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10
```

## How It Works

1. **Data Ingestion**: Continuously monitors Aerodrome protocol via QuickNode API
2. **Memory Storage**: Stores observations in Mem0 with confidence scores
3. **Pattern Recognition**: Identifies patterns and correlations in protocol behavior
4. **Confidence Adjustment**: Validates predictions and adjusts confidence scores
5. **Memory Pruning**: Removes low-confidence and outdated information
6. **Intelligence Generation**: Uses Gemini to generate insights from high-confidence memories
7. **API Response**: Provides confidence-scored answers to queries

## Part 2: Profit Generation (Separate System)

The brain provides intelligence that can be used by separate trading systems for:
- Arbitrage opportunity identification
- Optimal voting strategies
- Liquidity provision timing
- Risk assessment and management

**Note**: The brain itself does not execute trades or manage funds. It provides intelligence for informed decision-making.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

MIT