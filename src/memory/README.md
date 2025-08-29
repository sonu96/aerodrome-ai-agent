# Memory Pruning Engine

A sophisticated memory pruning engine for the Aerodrome AI Agent that leverages Mem0's latest features including graph memory, v2 search API, batch operations, and metadata filtering.

## Features

### Core Components

1. **Memory Categories** (`memory_categories.py`)
   - 10 specialized memory categories for Aerodrome-specific data
   - Category-specific retention policies and decay rates
   - Tiered pruning strategies (hourly, daily, weekly, monthly)
   - Intelligent thresholds based on memory importance

2. **Enhanced Mem0 Client** (`mem0_client.py`)
   - Async operations with connection pooling
   - Graph memory support (Neo4j integration)
   - Batch operations for high-performance processing
   - Advanced filtering with metadata queries
   - Memory export/import in multiple formats (JSON, CSV, JSONL)
   - Comprehensive error handling and retry logic

3. **Pruning Engine** (`pruning_engine.py`)
   - Multi-tiered pruning strategies
   - Memory consolidation based on similarity analysis
   - Intelligent cleanup using confidence scores and aging
   - Graph memory pruning for Neo4j relationships
   - Parallel processing for large datasets
   - Storage optimization and defragmentation

### Memory Categories

| Category | Retention Period | Decay Rate | Description |
|----------|-----------------|------------|-------------|
| `PROTOCOL_CONSTANTS` | 365 days | 0.001 | Stable protocol information |
| `POOL_PERFORMANCE` | 90 days | 0.01 | Pool metrics and performance data |
| `VOTING_PATTERNS` | 180 days | 0.005 | Governance voting behaviors |
| `MARKET_CORRELATIONS` | 60 days | 0.02 | Market relationship analysis |
| `SPECULATIVE_INSIGHTS` | 30 days | 0.05 | Short-term speculative content |
| `USER_PREFERENCES` | 365 days | 0.002 | User settings and preferences |
| `TRADING_STRATEGIES` | 120 days | 0.008 | Trading approach documentation |
| `RISK_ASSESSMENTS` | 45 days | 0.015 | Risk analysis and scoring |
| `HISTORICAL_EVENTS` | 720 days | 0.0005 | Long-term historical data |
| `TECHNICAL_ANALYSIS` | 30 days | 0.03 | Technical indicators and patterns |

### Pruning Strategies

#### 1. Tiered Pruning
- **Hourly**: Removes speculative insights and technical analysis
- **Daily**: Processes market correlations, pool performance, and risk assessments
- **Weekly**: Handles voting patterns and trading strategies  
- **Monthly**: Manages user preferences, protocol constants, and historical events

#### 2. Confidence-Based Pruning
- Removes memories below category-specific confidence thresholds
- Considers access frequency and importance scores
- Applies decay factors based on memory age

#### 3. Memory Consolidation
- Groups similar memories using Jaccard similarity analysis
- Combines redundant information into consolidated entries
- Preserves highest confidence and most recent data

#### 4. Graph Memory Pruning
- Identifies orphaned nodes with broken relationships
- Removes memories with invalid graph connections
- Maintains graph consistency and integrity

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export MEM0_API_KEY="your_mem0_api_key"
export NEO4J_URL="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your_password"
```

## Usage

### Basic Setup

```python
import asyncio
from src.memory import EnhancedMem0Client, MemoryPruningEngine, MemoryCategory

async def main():
    # Initialize client
    client = EnhancedMem0Client(
        api_key="your_api_key",
        enable_graph=True,
        neo4j_config={
            "url": "bolt://localhost:7687",
            "username": "neo4j", 
            "password": "password"
        }
    )
    
    # Add memory
    await client.add_memory(
        content="ETH-USDC pool showing 15% APY",
        category=MemoryCategory.POOL_PERFORMANCE,
        confidence=0.85,
        metadata={"apy": 15.0, "pool_tvl": 1000000}
    )
    
    # Initialize pruning engine
    engine = MemoryPruningEngine(client)
    
    # Run pruning
    results = await engine.run_tiered_pruning(
        force_tiers=["daily"],
        dry_run=False
    )
```

### Advanced Features

#### Batch Operations
```python
# Batch add memories
memories = [
    {"content": "Pool A performance", "confidence": 0.8},
    {"content": "Pool B analysis", "confidence": 0.75}
]

results = await client.batch_add_memories(
    memories=memories,
    category=MemoryCategory.POOL_PERFORMANCE
)
```

#### Advanced Search
```python
# Search with filters
results = await client.search_memories(
    query="correlation analysis",
    category=MemoryCategory.MARKET_CORRELATIONS,
    confidence_threshold=0.7,
    filters={
        "correlation_strength": {"$gte": 0.8},
        "timeframe": {"$in": ["30d", "7d"]}
    }
)
```

#### Memory Export/Import
```python
# Export memories
export_data = await client.export_memories(
    category=MemoryCategory.POOL_PERFORMANCE,
    format_type="json"
)

# Import memories
import_results = await client.import_memories(
    data=export_data,
    format_type="json"
)
```

#### Pruning Recommendations
```python
# Get recommendations without pruning
recommendations = await engine.get_pruning_recommendations()
print(f"Recommended deletions: {recommendations['recommended_deletions']}")
print(f"Recommended consolidations: {recommendations['recommended_consolidations']}")
```

## Configuration

### Engine Configuration
```python
config = {
    "similarity_threshold": 0.85,      # Consolidation similarity threshold
    "consolidation_enabled": True,      # Enable memory consolidation
    "graph_pruning_enabled": True,      # Enable graph-based pruning
    "parallel_processing": True,        # Use parallel processing
    "max_workers": 4,                   # Maximum worker threads
    "batch_size": 100                   # Default batch size
}

engine = MemoryPruningEngine(client, config=config)
```

### Category Policies
```python
from src.memory import MemoryCategoryConfig

# Get policy for a category
policy = MemoryCategoryConfig.get_policy(MemoryCategory.POOL_PERFORMANCE)
print(f"Max age: {policy.max_age}")
print(f"Decay rate: {policy.decay_rate}")
print(f"Batch size: {policy.batch_size}")

# Calculate decay factor
decay_factor = MemoryCategoryConfig.calculate_decay_factor(
    MemoryCategory.MARKET_CORRELATIONS,
    age_hours=72  # 3 days
)
```

## Monitoring

### Memory Statistics
```python
stats = await client.get_memory_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"Graph memories: {stats['graph_memories']}")
print(f"Categories: {stats['by_category']}")
```

### Pruning History
```python
history = engine.get_pruning_history(limit=10)
for entry in history:
    print(f"Deleted: {entry['memories_deleted']}")
    print(f"Consolidated: {entry['memories_consolidated']}")
    print(f"Deletion rate: {entry['deletion_rate']:.1f}%")
```

## Error Handling

The system includes comprehensive error handling:

- Automatic retry logic with exponential backoff
- Rate limiting protection
- Connection pooling and management
- Graceful degradation for partial failures
- Detailed error logging and tracking

## Performance Optimization

### Batch Processing
- Configurable batch sizes per memory category
- Parallel processing for large datasets
- Connection pooling for database operations
- Efficient memory consolidation algorithms

### Storage Optimization
- Metadata compression and deduplication
- Redundant field removal
- Timestamp optimization
- Graph relationship pruning

## Graph Memory Features

When using Neo4j integration:

```python
# Add memory with graph relationships
await client.add_memory(
    content="Pool performance correlates with token price",
    category=MemoryCategory.MARKET_CORRELATIONS,
    confidence=0.8,
    metadata={
        "relationships": ["pool_123", "token_abc"],
        "correlation_strength": 0.85
    },
    enable_graph=True
)

# Graph-based search
results = await client.search_memories(
    query="price correlation",
    use_graph=True,
    filters={"correlation_strength": {"$gte": 0.7}}
)
```

## API Reference

### EnhancedMem0Client Methods

- `add_memory()` - Add single memory with metadata
- `batch_add_memories()` - Add multiple memories efficiently
- `search_memories()` - Search with advanced filters
- `get_memories_by_category()` - Retrieve category-specific memories
- `update_memory()` - Update existing memory
- `delete_memory()` - Delete single memory
- `batch_delete_memories()` - Delete multiple memories
- `export_memories()` - Export in various formats
- `import_memories()` - Import from various formats
- `get_memory_stats()` - Get comprehensive statistics

### MemoryPruningEngine Methods

- `run_tiered_pruning()` - Execute tiered pruning strategies
- `optimize_memory_storage()` - Optimize storage and defragment
- `get_pruning_recommendations()` - Get pruning suggestions
- `get_pruning_history()` - Retrieve pruning operation history

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.