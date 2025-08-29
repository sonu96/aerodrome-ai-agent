# Aerodrome DeFi Agent Brain - Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Design Decisions](#design-decisions)

## Overview

The Aerodrome DeFi Agent Brain is an autonomous AI system designed for personal DeFi portfolio management on the Aerodrome Finance protocol. It combines advanced AI decision-making with blockchain interactions to optimize yield farming, liquidity provision, and governance participation.

### Key Features
- **Fully Autonomous**: Operates 24/7 without manual intervention
- **Memory-Driven**: Learns from past actions and market patterns
- **CDP SDK Native**: All blockchain operations through Coinbase Developer Platform SDK
- **Single User Optimized**: Streamlined for personal use without unnecessary auth layers

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface Layer                         │
│                  (CLI / API / Monitoring Dashboard)              │
└─────────────────────────────────────────────────────────────────┘
                              ↕
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
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer (GCP)                    │
│        (Cloud Run / Firestore / BigQuery / Secret Manager)       │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Brain Core (LangGraph State Machine)
The brain operates as a directed graph with the following nodes:

- **Observe Node**: Collects market data and wallet state
- **Recall Node**: Retrieves relevant memories and patterns
- **Analyze Node**: Identifies opportunities and calculates scores
- **Decide Node**: Makes risk-adjusted decisions
- **Execute Node**: Performs blockchain operations via CDP SDK
- **Learn Node**: Extracts patterns from results
- **Prune Node**: Manages memory optimization

### 2. CDP SDK Integration Layer
All blockchain interactions are exclusively handled through CDP SDK:

- **Wallet Management**: MPC wallets with 2-of-2 security
- **Smart Contract Operations**: Direct contract invocation
- **Event Monitoring**: Real-time blockchain event tracking
- **Transaction Building**: Gas optimization and safety checks

### 3. Memory System (Mem0)
Intelligent memory management with automatic pruning:

- **Multi-tier Storage**: Hot → Warm → Cold → Archive
- **Pattern Extraction**: Automatic pattern recognition from repeated actions
- **Smart Pruning**: Age, relevance, and redundancy-based cleanup
- **Compression**: Similar memories compressed into patterns

### 4. Decision Engine
Multi-factor decision system:

- **Opportunity Scoring**: Market conditions + historical patterns
- **Risk Assessment**: Position sizing + failure pattern matching
- **Confidence Calculation**: Combined score with safety thresholds
- **Execution Planning**: Transaction parameter optimization

## Data Flow

### Standard Operation Cycle
```
1. Market Observation
   ↓
2. Memory Recall (relevant patterns)
   ↓
3. Opportunity Analysis
   ↓
4. Risk-Adjusted Decision
   ↓
5. CDP SDK Execution
   ↓
6. Result Learning
   ↓
7. Memory Pruning
   ↓
8. Cycle Repeat (60s interval)
```

### Emergency Flow
```
Risk Threshold Exceeded
   ↓
Emergency Stop Triggered
   ↓
Position Liquidation
   ↓
Alert User
   ↓
Safe Mode Activation
```

## Technology Stack

### Core Technologies
- **Python 3.11+**: Primary language
- **LangGraph**: State machine orchestration
- **CDP SDK**: Blockchain interactions
- **Mem0**: Memory management
- **OpenAI GPT-4**: LLM reasoning

### Infrastructure
- **GCP Cloud Run**: Main brain service
- **Firestore**: Real-time state storage
- **BigQuery**: Historical data analytics
- **Secret Manager**: Secure key storage
- **Cloud Scheduler**: Periodic tasks

### Blockchain
- **Base Network**: Primary chain (Aerodrome native)
- **Aerodrome Protocol**: Target DeFi protocol
- **CDP MPC Wallets**: Secure key management

## Design Decisions

### Why CDP SDK Only?
1. **Security**: MPC wallets eliminate single points of failure
2. **Simplicity**: Unified interface for all blockchain operations
3. **Reliability**: Built-in retry logic and error handling
4. **Type Safety**: Automatic ABI type inference

### Why LangGraph?
1. **State Management**: Clear state transitions and routing
2. **Modularity**: Each node handles specific responsibility
3. **Testability**: Individual nodes can be tested in isolation
4. **Observability**: Built-in execution tracing

### Why Mem0 with Pruning?
1. **Scalability**: Prevents unbounded memory growth
2. **Relevance**: Keeps only valuable patterns
3. **Performance**: Tiered storage optimizes access speed
4. **Cost**: Reduces storage costs through compression

### Single User Optimization
1. **No Auth Overhead**: Direct API key usage
2. **Simplified State**: No multi-tenancy concerns
3. **Faster Decisions**: No permission checks
4. **Personal Learning**: Memories optimized for one user's patterns

## Security Considerations

### Key Management
- CDP MPC wallets (2-of-2) for production
- GCP Secret Manager for API keys
- Environment variables for development
- No keys in code or logs

### Transaction Safety
- Pre-execution simulation
- Slippage protection
- Gas price limits
- Emergency stop mechanisms

### Data Protection
- Encrypted storage at rest
- TLS for all communications
- No sensitive data in logs
- Regular security audits

## Performance Targets

### Response Times
- Market observation: < 1 second
- Decision making: < 2 seconds
- Transaction execution: < 5 seconds
- Memory recall: < 500ms

### Reliability
- 99.9% uptime target
- Automatic failover
- Graceful degradation
- Self-healing capabilities

## Monitoring & Observability

### Key Metrics
- Decision confidence scores
- Execution success rate
- Memory usage and pruning efficiency
- Pattern extraction quality

### Alerting
- Failed transactions
- Low confidence decisions
- Memory threshold exceeded
- Unusual market conditions

## Future Enhancements

### Planned Features
1. Multi-chain support (Ethereum, Arbitrum)
2. Advanced ML models for pattern recognition
3. Natural language command interface
4. Mobile monitoring app

### Scalability Path
1. Distributed processing for multiple strategies
2. Parallel execution capabilities
3. Advanced caching layers
4. Real-time streaming data processing