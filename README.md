# Aerodrome AI Agent

An autonomous AI-powered DeFi portfolio manager for Aerodrome Finance protocol on Base network. This agent uses advanced AI decision-making with LangGraph state machines, Mem0 memory systems, and Coinbase Developer Platform (CDP) SDK for secure blockchain operations.

## Features

- **Fully Autonomous**: Operates 24/7 without manual intervention
- **Memory-Driven**: Learns from past actions and market patterns using Mem0
- **CDP SDK Native**: All blockchain operations through Coinbase Developer Platform SDK
- **LangGraph Brain**: Sophisticated state machine for decision-making
- **Risk Management**: Built-in safety mechanisms and emergency stops
- **Pattern Recognition**: Extracts and learns from trading patterns
- **Multi-tier Memory**: Intelligent memory management with automatic pruning

## Architecture

The agent is built with a modular architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestration Layer                 │
│            (State Machine / Decision Flow / Routing)             │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                      Cognitive Processing Layer                  │
│        (Analysis / Strategy / Risk Assessment / Learning)        │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                        CDP SDK Layer                             │
│    (Wallet Management / Smart Contracts / Transactions)          │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                         Memory Layer (Mem0)                      │
│      (Pattern Storage / Learning / Pruning / Compression)        │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

- **Brain Core**: LangGraph-based state machine with cognitive nodes
- **Memory System**: Mem0-powered learning and pattern recognition  
- **CDP Integration**: Secure blockchain operations via CDP SDK
- **Contract Layer**: Aerodrome protocol contract interfaces
- **Configuration**: Environment-based configuration management

## Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Coinbase Developer Platform API keys
- Qdrant vector database (for memory system)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aerodrome-ai/agent.git
   cd aerodrome-ai-agent
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Install Qdrant (for memory system)**:
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or install locally
   # See: https://qdrant.tech/documentation/guides/installation/
   ```

### Configuration

Edit the `.env` file with your configuration:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
CDP_API_KEY_NAME=your_cdp_api_key_name
CDP_API_KEY_PRIVATE_KEY=your_cdp_private_key_here

# Network (use base-sepolia for testing)
NETWORK=base-mainnet
CDP_NETWORK_ID=base-mainnet

# Agent Configuration
OPERATION_MODE=autonomous  # or simulation for testing
RISK_LEVEL=medium
BRAIN_CONFIDENCE_THRESHOLD=0.7
```

### Running the Agent

1. **Development Mode** (with logging):
   ```bash
   python -m aerodrome_ai_agent.cli start --mode development
   ```

2. **Simulation Mode** (safe testing):
   ```bash
   python -m aerodrome_ai_agent.cli start --mode simulation
   ```

3. **Production Mode**:
   ```bash
   python -m aerodrome_ai_agent.cli start --mode autonomous
   ```

### Monitoring

The agent provides monitoring endpoints:

- **Health Check**: http://localhost:8081/health
- **Metrics**: http://localhost:8080/metrics
- **Status**: http://localhost:8080/status

## Usage Examples

### Basic Usage

```python
from aerodrome_ai_agent import AerodromeBrain, AgentConfig
from aerodrome_ai_agent.memory import MemorySystem
from aerodrome_ai_agent.cdp import CDPManager

# Initialize components
config = AgentConfig.from_env()
memory_system = MemorySystem(config.get_memory_config())
cdp_manager = CDPManager(config.get_cdp_config())

# Create and start brain
brain = AerodromeBrain(
    config=config.get_brain_config(),
    memory_system=memory_system,
    cdp_manager=cdp_manager
)

# Run a single cycle
result = await brain.run_cycle()

# Or start continuous operation
await brain.start_continuous_operation()
```

### Memory System

```python
from aerodrome_ai_agent.memory import MemorySystem, MemoryConfig

memory = MemorySystem(MemoryConfig.from_env())

# Learn from experience
experience = {
    "action_type": "ADD_LIQUIDITY",
    "pool": "WETH-USDC",
    "amount": 1000
}
outcome = {
    "success": True,
    "profit": 25.0,
    "confidence": 0.85
}

memory_id = await memory.learn_from_experience(experience, outcome)

# Recall relevant memories
context = {"action_type": "ADD_LIQUIDITY", "pool": "WETH-USDC"}
memories = await memory.recall_relevant_memories(context)

# Extract patterns
patterns = await memory.extract_patterns()
```

### CDP Integration

```python
from aerodrome_ai_agent.cdp import CDPManager, CDPConfig

cdp = CDPManager(CDPConfig.from_env())

# Initialize wallet
await cdp.initialize_wallet()

# Get balances
balances = await cdp.get_balances(["eth", "usdc", "aero"])

# Read contract
pool_data = await cdp.read_contract(
    contract_address="0x...",
    method="getReserves",
    abi=POOL_ABI
)
```

## Development

### Project Structure

```
aerodrome-ai-agent/
├── src/
│   └── aerodrome_ai_agent/
│       ├── brain/           # LangGraph brain implementation
│       │   ├── core.py      # Main brain class
│       │   ├── config.py    # Brain configuration
│       │   ├── state.py     # State definitions
│       │   └── nodes/       # Individual cognitive nodes
│       ├── memory/          # Mem0 memory system
│       │   ├── system.py    # Main memory class
│       │   ├── config.py    # Memory configuration
│       │   ├── patterns.py  # Pattern extraction
│       │   └── pruning.py   # Memory pruning
│       ├── cdp/             # CDP SDK integration
│       │   ├── manager.py   # Main CDP interface
│       │   ├── config.py    # CDP configuration
│       │   ├── wallet.py    # Wallet management
│       │   └── contracts.py # Contract interactions
│       ├── contracts/       # Aerodrome contract ABIs
│       │   ├── abis.py      # Contract ABIs
│       │   ├── addresses.py # Contract addresses
│       │   └── utils.py     # Contract utilities
│       ├── config/          # Configuration management
│       │   ├── base.py      # Base configuration
│       │   └── settings.py  # Environment settings
│       └── utils/           # Shared utilities
├── tests/                   # Test files
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
└── .env.example            # Environment template
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aerodrome_ai_agent

# Run specific test module
pytest tests/test_brain.py
```

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy .

# Linting
ruff check .
```

## Safety Features

### Risk Management

- **Position Size Limits**: Maximum 20% of portfolio per position
- **Slippage Protection**: Maximum 2% slippage tolerance
- **Emergency Stop**: Automatic emergency stop at 10% portfolio loss
- **Gas Price Limits**: Maximum gas price protection
- **Transaction Simulation**: All transactions simulated before execution

### Memory Safety

- **Automatic Pruning**: Old and irrelevant memories automatically removed
- **Pattern Validation**: Only high-confidence patterns used for decisions
- **Memory Limits**: Maximum memory usage caps prevent resource exhaustion
- **Quality Filters**: Low-quality memories filtered out

### Security

- **CDP MPC Wallets**: Secure multi-party computation wallets
- **No Private Keys**: Private keys managed by CDP, never exposed
- **API Key Protection**: Secure API key management
- **Transaction Verification**: All transactions verified before execution

## Configuration Reference

### Brain Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BRAIN_CONFIDENCE_THRESHOLD` | 0.7 | Minimum confidence for execution |
| `BRAIN_RISK_THRESHOLD` | 0.3 | Maximum acceptable risk level |
| `BRAIN_MIN_OPPORTUNITY_SCORE` | 0.6 | Minimum opportunity score |
| `BRAIN_MAX_POSITION_SIZE` | 0.2 | Maximum position size (20%) |
| `BRAIN_MAX_SLIPPAGE` | 0.02 | Maximum slippage tolerance |
| `BRAIN_EMERGENCY_STOP_LOSS` | 0.1 | Emergency stop loss threshold |

### Memory Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MEMORY_MAX_MEMORIES` | 10000 | Maximum stored memories |
| `MEMORY_MAX_AGE_DAYS` | 90 | Maximum memory age |
| `MEMORY_PATTERN_THRESHOLD` | 5 | Min occurrences for pattern |
| `MEMORY_PRUNING_ENABLED` | true | Enable automatic pruning |
| `MEMORY_HOT_TIER_DAYS` | 7 | Hot tier memory duration |

### CDP Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CDP_MAX_GAS_PRICE` | 50.0 | Maximum gas price (gwei) |
| `CDP_SIMULATION_REQUIRED` | true | Require transaction simulation |
| `CDP_MAX_RETRIES` | 3 | Maximum retry attempts |
| `CDP_TRANSACTION_TIMEOUT` | 300 | Transaction timeout (seconds) |

## Monitoring and Observability

### Metrics

The agent exposes Prometheus metrics:

- **Decision metrics**: Confidence scores, success rates
- **Execution metrics**: Transaction success, gas usage
- **Memory metrics**: Memory usage, pattern extraction
- **Performance metrics**: Cycle times, error rates

### Logging

Structured logging with different levels:

- **DEBUG**: Detailed execution information
- **INFO**: General operational information
- **WARNING**: Non-critical issues
- **ERROR**: Error conditions
- **CRITICAL**: Critical system failures

### Health Checks

Health check endpoint returns:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T00:00:00Z",
  "components": {
    "brain": "healthy",
    "memory": "healthy", 
    "cdp": "healthy"
  },
  "metrics": {
    "cycles_completed": 1000,
    "success_rate": 0.85,
    "avg_profit": 12.5
  }
}
```

## Troubleshooting

### Common Issues

1. **CDP Connection Errors**
   - Check API keys in `.env`
   - Verify network configuration
   - Ensure wallet has sufficient funds

2. **Memory System Issues**
   - Check Qdrant is running
   - Verify connection settings
   - Check disk space for vector database

3. **Low Decision Confidence**
   - Review market conditions
   - Check memory system for relevant patterns
   - Adjust confidence threshold if needed

4. **Transaction Failures**
   - Check gas price settings
   - Verify slippage tolerance
   - Check wallet balance and approvals

### Debug Mode

Run with debug logging:

```bash
DEBUG=true LOG_LEVEL=DEBUG python -m aerodrome_ai_agent.cli start
```

### Support

- **Documentation**: [https://docs.aerodrome-agent.ai](https://docs.aerodrome-agent.ai)
- **Issues**: [GitHub Issues](https://github.com/aerodrome-ai/agent/issues)
- **Discord**: [Aerodrome AI Community](https://discord.gg/aerodrome-ai)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes. Use at your own risk. The authors are not responsible for any financial losses. Always test thoroughly before using with real funds.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Built with ❤️ for the Aerodrome Finance ecosystem**