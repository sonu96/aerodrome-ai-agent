# Memory System Documentation

## Table of Contents
1. [Overview](#overview)
2. [Memory Architecture](#memory-architecture)  
3. [Mem0 Integration](#mem0-integration)
4. [Memory Categories](#memory-categories)
5. [Pruning Strategies](#pruning-strategies)
6. [Pattern Extraction](#pattern-extraction)
7. [Storage Tiers](#storage-tiers)
8. [Memory Operations](#memory-operations)
9. [Performance Optimization](#performance-optimization)

## Overview

The memory system is the brain's learning and adaptation mechanism, built on Mem0 with intelligent pruning and pattern extraction. It enables the agent to learn from past experiences, recognize patterns, and make increasingly better decisions over time.

### Key Features
- **Multi-tier Storage**: Hot → Warm → Cold → Archive with automatic migration
- **Intelligent Pruning**: Age, relevance, and redundancy-based cleanup
- **Pattern Extraction**: Automatic pattern recognition from repeated behaviors
- **Compression**: Similar memories compressed into meta-patterns
- **Single User Optimization**: Streamlined for personal use without multi-tenancy overhead

## Memory Architecture

### Core Memory System

```python
from mem0 import Memory
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import DBSCAN
import hashlib
import json

@dataclass
class MemoryConfig:
    """Configuration for the memory system"""
    
    # Storage limits
    max_hot_memories: int = 1000
    max_warm_memories: int = 5000
    max_cold_memories: int = 10000
    max_patterns: int = 500
    
    # Time thresholds (in hours)
    hot_threshold: int = 24
    warm_threshold: int = 168  # 7 days
    cold_threshold: int = 720  # 30 days
    
    # Relevance thresholds
    min_relevance_score: float = 0.3
    pattern_threshold: int = 5  # Min occurrences for pattern
    
    # Compression settings
    compression_ratio: float = 0.7
    similarity_threshold: float = 0.85

class MemorySystem:
    """Advanced memory system with intelligent management"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        
        # Initialize Mem0
        self.mem0 = self._initialize_mem0()
        
        # Memory tiers
        self.tiers = {
            'hot': {},     # Recent, frequently accessed
            'warm': {},    # Older, occasionally accessed
            'cold': {},    # Old, rarely accessed
            'archive': {}  # Patterns only
        }
        
        # Pattern storage
        self.patterns = {}
        self.pattern_index = {}
        
        # Access tracking
        self.access_counts = {}
        self.last_access = {}
        
    def _initialize_mem0(self) -> Memory:
        """Initialize Mem0 with configuration"""
        
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "model": "gpt-4-turbo",
                    "temperature": 0.1  # Low temperature for consistency
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "model": "text-embedding-3-small"
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "aerodrome_brain",
                    "host": "localhost",
                    "port": 6333,
                    "embedding_model_dims": 1536
                }
            },
            "history_db_path": "./memory_history.db",
            "version": "v1.1"
        }
        
        return Memory.from_config(config)
```

## Mem0 Integration

### Memory Storage and Retrieval

```python
class Mem0Operations:
    """Core Mem0 operations wrapper"""
    
    async def add_memory(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add memory to Mem0 with metadata"""
        
        # Generate unique ID
        memory_id = self._generate_memory_id(content)
        
        # Prepare memory content
        memory_text = self._format_memory(content)
        
        # Add metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'id': memory_id,
            'timestamp': datetime.now().isoformat(),
            'type': content.get('type', 'general'),
            'category': self._categorize_memory(content),
            'ttl': self._calculate_ttl(content),
            'access_count': 0,
            'tier': 'hot'
        })
        
        # Store in Mem0
        self.mem0.add(
            memory_text,
            user_id="me",  # Single user
            metadata=metadata
        )
        
        # Track in local tier
        self.tiers['hot'][memory_id] = {
            'content': content,
            'metadata': metadata,
            'embedding': None  # Will be set by Mem0
        }
        
        return memory_id
    
    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search memories using Mem0's semantic search"""
        
        # Search in Mem0
        results = self.mem0.search(
            query=query,
            user_id="me",
            limit=limit * 2  # Get more for filtering
        )
        
        # Apply additional filters
        if filters:
            results = self._apply_filters(results, filters)
        
        # Update access tracking
        for result in results:
            memory_id = result['metadata'].get('id')
            if memory_id:
                self.access_counts[memory_id] = self.access_counts.get(memory_id, 0) + 1
                self.last_access[memory_id] = datetime.now()
        
        # Sort by relevance and recency
        results = self._rank_results(results, query)
        
        return results[:limit]
    
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update existing memory"""
        
        # Get current memory
        current = self._get_memory_by_id(memory_id)
        if not current:
            return False
        
        # Merge updates
        updated_content = {**current['content'], **updates}
        updated_metadata = current['metadata'].copy()
        updated_metadata['last_modified'] = datetime.now().isoformat()
        updated_metadata['modification_count'] = updated_metadata.get('modification_count', 0) + 1
        
        # Update in Mem0
        self.mem0.update(
            memory_id=memory_id,
            text=self._format_memory(updated_content),
            metadata=updated_metadata
        )
        
        # Update in tier
        tier = updated_metadata.get('tier', 'hot')
        if memory_id in self.tiers[tier]:
            self.tiers[tier][memory_id] = {
                'content': updated_content,
                'metadata': updated_metadata
            }
        
        return True
    
    def _generate_memory_id(self, content: Dict) -> str:
        """Generate unique memory ID"""
        
        # Create hash from content
        content_str = json.dumps(content, sort_keys=True)
        hash_obj = hashlib.sha256(content_str.encode())
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().isoformat()
        
        return f"mem_{hash_obj.hexdigest()[:12]}_{int(datetime.now().timestamp())}"
    
    def _format_memory(self, content: Dict) -> str:
        """Format memory content for storage"""
        
        memory_type = content.get('type', 'general')
        
        if memory_type == 'trade':
            return self._format_trade_memory(content)
        elif memory_type == 'market_observation':
            return self._format_market_memory(content)
        elif memory_type == 'pattern':
            return self._format_pattern_memory(content)
        else:
            return json.dumps(content)
    
    def _format_trade_memory(self, content: Dict) -> str:
        """Format trade memory"""
        
        return (
            f"Trade: {content['action']} "
            f"{content.get('amount', 'N/A')} {content.get('token', 'N/A')} "
            f"in pool {content.get('pool', 'N/A')}. "
            f"Result: {content.get('result', 'pending')}. "
            f"Profit: {content.get('profit', 0)}. "
            f"Gas: {content.get('gas_used', 0)}."
        )
```

## Memory Categories

### Category Definitions

```python
class MemoryCategories:
    """Define and manage memory categories"""
    
    CATEGORIES = {
        'trades': {
            'description': 'Trade executions and results',
            'retention': 30,  # days
            'importance': 0.9
        },
        'market_observations': {
            'description': 'Market state snapshots',
            'retention': 7,
            'importance': 0.6
        },
        'patterns': {
            'description': 'Learned patterns and strategies',
            'retention': 365,
            'importance': 1.0
        },
        'failures': {
            'description': 'Failed actions and errors',
            'retention': 60,
            'importance': 0.8
        },
        'opportunities': {
            'description': 'Identified opportunities',
            'retention': 3,
            'importance': 0.5
        },
        'user_preferences': {
            'description': 'User settings and preferences',
            'retention': -1,  # Never delete
            'importance': 1.0
        }
    }
    
    def categorize_memory(self, content: Dict) -> str:
        """Categorize memory based on content"""
        
        memory_type = content.get('type', '')
        
        # Direct mapping
        type_to_category = {
            'trade': 'trades',
            'swap': 'trades',
            'liquidity': 'trades',
            'observation': 'market_observations',
            'pattern': 'patterns',
            'error': 'failures',
            'opportunity': 'opportunities',
            'preference': 'user_preferences'
        }
        
        for key, category in type_to_category.items():
            if key in memory_type.lower():
                return category
        
        # Default category
        return 'market_observations'
    
    def get_retention_period(self, category: str) -> int:
        """Get retention period for category in days"""
        
        return self.CATEGORIES.get(category, {}).get('retention', 30)
    
    def get_importance(self, category: str) -> float:
        """Get importance score for category"""
        
        return self.CATEGORIES.get(category, {}).get('importance', 0.5)
```

### Specialized Memory Types

```python
class TradeMemory:
    """Specialized memory for trades"""
    
    @dataclass
    class TradeData:
        action: str  # swap, add_liquidity, remove_liquidity
        pool: str
        tokens: List[str]
        amounts: List[float]
        timestamp: datetime
        gas_used: float
        success: bool
        profit: float
        market_conditions: Dict
        decision_confidence: float
    
    def create_trade_memory(self, trade: TradeData) -> Dict:
        """Create structured trade memory"""
        
        return {
            'type': 'trade',
            'action': trade.action,
            'pool': trade.pool,
            'tokens': trade.tokens,
            'amounts': trade.amounts,
            'timestamp': trade.timestamp.isoformat(),
            'gas_used': trade.gas_used,
            'success': trade.success,
            'profit': trade.profit,
            'market_conditions': trade.market_conditions,
            'decision_confidence': trade.decision_confidence,
            'category': 'trades',
            'searchable_text': self._create_searchable_text(trade)
        }
    
    def _create_searchable_text(self, trade: TradeData) -> str:
        """Create searchable text representation"""
        
        return (
            f"{trade.action} {' '.join(trade.tokens)} "
            f"pool {trade.pool} "
            f"{'successful' if trade.success else 'failed'} "
            f"profit {trade.profit}"
        )

class PatternMemory:
    """Specialized memory for patterns"""
    
    @dataclass
    class PatternData:
        pattern_type: str
        conditions: Dict
        action_sequence: List[str]
        success_rate: float
        occurrence_count: int
        avg_profit: float
        confidence: float
        discovered_at: datetime
    
    def create_pattern_memory(self, pattern: PatternData) -> Dict:
        """Create structured pattern memory"""
        
        return {
            'type': 'pattern',
            'pattern_type': pattern.pattern_type,
            'conditions': pattern.conditions,
            'action_sequence': pattern.action_sequence,
            'success_rate': pattern.success_rate,
            'occurrence_count': pattern.occurrence_count,
            'avg_profit': pattern.avg_profit,
            'confidence': pattern.confidence,
            'discovered_at': pattern.discovered_at.isoformat(),
            'category': 'patterns',
            'importance': pattern.success_rate * pattern.confidence
        }
```

## Pruning Strategies

### Multi-Level Pruning System

```python
class MemoryPruning:
    """Intelligent memory pruning system"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.pruning_stats = {
            'total_pruned': 0,
            'by_age': 0,
            'by_relevance': 0,
            'by_redundancy': 0,
            'patterns_extracted': 0
        }
    
    async def execute_pruning_cycle(self) -> Dict[str, Any]:
        """Execute complete pruning cycle"""
        
        start_time = datetime.now()
        initial_count = self._get_total_memory_count()
        
        # 1. Age-based pruning
        age_pruned = await self.age_based_pruning()
        
        # 2. Relevance-based pruning
        relevance_pruned = await self.relevance_based_pruning()
        
        # 3. Redundancy pruning
        redundancy_pruned = await self.redundancy_pruning()
        
        # 4. Pattern extraction before final pruning
        patterns = await self.extract_patterns()
        
        # 5. Tier migration
        await self.migrate_tiers()
        
        # 6. Compression
        compressed = await self.compress_similar_memories()
        
        # Calculate statistics
        final_count = self._get_total_memory_count()
        
        return {
            'duration': (datetime.now() - start_time).total_seconds(),
            'initial_count': initial_count,
            'final_count': final_count,
            'pruned': {
                'by_age': age_pruned,
                'by_relevance': relevance_pruned,
                'by_redundancy': redundancy_pruned
            },
            'patterns_extracted': len(patterns),
            'compressed': compressed,
            'reduction_ratio': 1 - (final_count / initial_count) if initial_count > 0 else 0
        }
    
    async def age_based_pruning(self) -> int:
        """Prune memories based on age and tier"""
        
        pruned_count = 0
        current_time = datetime.now()
        
        for tier_name, tier_memories in self.memory.tiers.items():
            if tier_name == 'archive':
                continue  # Don't prune archives
            
            for memory_id, memory_data in list(tier_memories.items()):
                metadata = memory_data.get('metadata', {})
                timestamp = datetime.fromisoformat(metadata.get('timestamp', ''))
                age_hours = (current_time - timestamp).total_seconds() / 3600
                
                should_prune = False
                
                if tier_name == 'hot' and age_hours > self.memory.config.hot_threshold:
                    # Move to warm or prune
                    if self._is_valuable(memory_data):
                        await self._move_to_tier(memory_id, 'hot', 'warm')
                    else:
                        should_prune = True
                        
                elif tier_name == 'warm' and age_hours > self.memory.config.warm_threshold:
                    # Move to cold or prune
                    if self._is_valuable(memory_data):
                        await self._move_to_tier(memory_id, 'warm', 'cold')
                    else:
                        should_prune = True
                        
                elif tier_name == 'cold' and age_hours > self.memory.config.cold_threshold:
                    # Extract pattern or prune
                    if self._can_extract_pattern(memory_data):
                        await self._extract_and_archive(memory_data)
                    should_prune = True
                
                if should_prune:
                    await self._prune_memory(memory_id)
                    pruned_count += 1
        
        self.pruning_stats['by_age'] += pruned_count
        return pruned_count
    
    async def relevance_based_pruning(self) -> int:
        """Prune based on relevance and access patterns"""
        
        pruned_count = 0
        
        for tier_name, tier_memories in self.memory.tiers.items():
            if tier_name in ['archive', 'hot']:
                continue  # Don't prune archives or hot memories
            
            for memory_id, memory_data in list(tier_memories.items()):
                relevance_score = self._calculate_relevance(memory_data)
                
                if relevance_score < self.memory.config.min_relevance_score:
                    # Check access pattern
                    access_count = self.memory.access_counts.get(memory_id, 0)
                    last_access = self.memory.last_access.get(memory_id)
                    
                    # If rarely accessed and low relevance, prune
                    if access_count < 2:
                        if last_access is None or (datetime.now() - last_access).days > 7:
                            await self._prune_memory(memory_id)
                            pruned_count += 1
        
        self.pruning_stats['by_relevance'] += pruned_count
        return pruned_count
    
    async def redundancy_pruning(self) -> int:
        """Remove redundant and duplicate memories"""
        
        pruned_count = 0
        
        # Group memories by similarity
        memory_clusters = await self._cluster_similar_memories()
        
        for cluster in memory_clusters:
            if len(cluster) > 1:
                # Keep the most valuable memory in cluster
                sorted_cluster = sorted(
                    cluster,
                    key=lambda m: self._calculate_memory_value(m),
                    reverse=True
                )
                
                # Keep best, prune rest
                for memory in sorted_cluster[1:]:
                    memory_id = memory['metadata']['id']
                    await self._prune_memory(memory_id)
                    pruned_count += 1
        
        self.pruning_stats['by_redundancy'] += pruned_count
        return pruned_count
    
    def _calculate_relevance(self, memory_data: Dict) -> float:
        """Calculate relevance score for memory"""
        
        score = 0.5  # Base score
        
        metadata = memory_data.get('metadata', {})
        content = memory_data.get('content', {})
        
        # Category importance
        category = metadata.get('category', 'general')
        category_importance = MemoryCategories().get_importance(category)
        score *= category_importance
        
        # Success factor (for trades)
        if content.get('success', False):
            score *= 1.2
        
        # Profit factor
        profit = content.get('profit', 0)
        if profit > 0:
            score *= (1 + min(profit / 100, 0.5))  # Cap at 50% boost
        
        # Access frequency
        access_count = self.memory.access_counts.get(metadata.get('id'), 0)
        score *= (1 + min(access_count / 10, 0.3))  # Cap at 30% boost
        
        # Recency factor
        timestamp = datetime.fromisoformat(metadata.get('timestamp', ''))
        age_days = (datetime.now() - timestamp).days
        recency_factor = max(0.5, 1 - (age_days / 30))  # Decay over 30 days
        score *= recency_factor
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _cluster_similar_memories(self) -> List[List[Dict]]:
        """Cluster similar memories using embeddings"""
        
        all_memories = []
        embeddings = []
        
        # Collect all memories and their embeddings
        for tier_memories in self.memory.tiers.values():
            for memory_id, memory_data in tier_memories.items():
                all_memories.append(memory_data)
                # Get embedding from Mem0
                embedding = await self._get_embedding(memory_data)
                embeddings.append(embedding)
        
        if not embeddings:
            return []
        
        # Use DBSCAN for clustering
        embeddings_array = np.array(embeddings)
        clustering = DBSCAN(eps=0.15, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings_array)
        
        # Group memories by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label != -1:  # Ignore noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(all_memories[idx])
        
        return list(clusters.values())
```

## Pattern Extraction

### Pattern Recognition System

```python
class PatternExtractor:
    """Extract patterns from memories"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.min_occurrences = 3
        self.confidence_threshold = 0.7
    
    async def extract_patterns(self) -> List[Dict]:
        """Extract patterns from memory history"""
        
        patterns = []
        
        # Extract different pattern types
        trade_patterns = await self.extract_trade_patterns()
        patterns.extend(trade_patterns)
        
        timing_patterns = await self.extract_timing_patterns()
        patterns.extend(timing_patterns)
        
        market_patterns = await self.extract_market_patterns()
        patterns.extend(market_patterns)
        
        # Store patterns
        for pattern in patterns:
            await self.memory.add_memory(
                content=pattern,
                metadata={'type': 'pattern', 'category': 'patterns'}
            )
        
        return patterns
    
    async def extract_trade_patterns(self) -> List[Dict]:
        """Extract patterns from trading history"""
        
        # Get all trade memories
        trades = await self.memory.search_memories(
            query="trade swap liquidity",
            limit=1000,
            filters={'category': 'trades'}
        )
        
        # Group by similar conditions
        pattern_groups = {}
        
        for trade in trades:
            content = trade.get('content', {})
            
            # Create pattern key from conditions
            pattern_key = self._create_pattern_key(content)
            
            if pattern_key not in pattern_groups:
                pattern_groups[pattern_key] = []
            
            pattern_groups[pattern_key].append(trade)
        
        # Extract patterns from groups
        patterns = []
        
        for pattern_key, group in pattern_groups.items():
            if len(group) >= self.min_occurrences:
                pattern = self._create_pattern_from_group(pattern_key, group)
                
                if pattern['confidence'] >= self.confidence_threshold:
                    patterns.append(pattern)
        
        return patterns
    
    def _create_pattern_key(self, trade_content: Dict) -> str:
        """Create pattern key from trade conditions"""
        
        # Extract key features
        features = {
            'action': trade_content.get('action'),
            'pool_type': 'stable' if trade_content.get('stable') else 'volatile',
            'market_condition': self._classify_market_condition(trade_content),
            'time_of_day': self._get_time_bucket(trade_content.get('timestamp'))
        }
        
        return json.dumps(features, sort_keys=True)
    
    def _create_pattern_from_group(
        self,
        pattern_key: str,
        group: List[Dict]
    ) -> Dict:
        """Create pattern from group of similar trades"""
        
        # Parse pattern key
        conditions = json.loads(pattern_key)
        
        # Calculate statistics
        successes = [t for t in group if t['content'].get('success', False)]
        success_rate = len(successes) / len(group)
        
        profits = [t['content'].get('profit', 0) for t in group]
        avg_profit = np.mean(profits)
        profit_std = np.std(profits)
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(
            len(group),
            success_rate,
            profit_std
        )
        
        return {
            'type': 'trade_pattern',
            'conditions': conditions,
            'statistics': {
                'occurrence_count': len(group),
                'success_rate': success_rate,
                'avg_profit': avg_profit,
                'profit_std': profit_std
            },
            'confidence': confidence,
            'discovered_at': datetime.now().isoformat(),
            'last_occurrence': max(t['metadata']['timestamp'] for t in group),
            'action_recommendation': self._get_action_recommendation(success_rate, avg_profit)
        }
    
    def _calculate_pattern_confidence(
        self,
        occurrence_count: int,
        success_rate: float,
        profit_std: float
    ) -> float:
        """Calculate confidence score for pattern"""
        
        # More occurrences = higher confidence
        occurrence_factor = min(occurrence_count / 10, 1.0)
        
        # Higher success rate = higher confidence
        success_factor = success_rate
        
        # Lower variance = higher confidence
        variance_factor = 1.0 / (1.0 + profit_std)
        
        confidence = (occurrence_factor * 0.3 + 
                     success_factor * 0.5 + 
                     variance_factor * 0.2)
        
        return confidence
```

## Storage Tiers

### Tier Management

```python
class StorageTiers:
    """Manage multi-tier storage system"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.tier_configs = {
            'hot': {
                'max_age_hours': 24,
                'max_size': 1000,
                'compression': None,
                'storage': 'memory'
            },
            'warm': {
                'max_age_hours': 168,
                'max_size': 5000,
                'compression': 'light',
                'storage': 'firestore'
            },
            'cold': {
                'max_age_hours': 720,
                'max_size': 10000,
                'compression': 'heavy',
                'storage': 'cloud_storage'
            },
            'archive': {
                'max_age_hours': -1,  # Infinite
                'max_size': -1,  # Unlimited
                'compression': 'pattern_only',
                'storage': 'bigquery'
            }
        }
    
    async def migrate_tiers(self) -> Dict[str, int]:
        """Migrate memories between tiers based on age and access"""
        
        migration_counts = {
            'hot_to_warm': 0,
            'warm_to_cold': 0,
            'cold_to_archive': 0
        }
        
        current_time = datetime.now()
        
        # Hot to Warm migration
        for memory_id, memory_data in list(self.memory.tiers['hot'].items()):
            age = self._get_memory_age(memory_data)
            
            if age > self.tier_configs['hot']['max_age_hours']:
                await self._migrate_memory(memory_id, 'hot', 'warm')
                migration_counts['hot_to_warm'] += 1
        
        # Warm to Cold migration
        for memory_id, memory_data in list(self.memory.tiers['warm'].items()):
            age = self._get_memory_age(memory_data)
            
            if age > self.tier_configs['warm']['max_age_hours']:
                await self._migrate_memory(memory_id, 'warm', 'cold')
                migration_counts['warm_to_cold'] += 1
        
        # Cold to Archive migration
        for memory_id, memory_data in list(self.memory.tiers['cold'].items()):
            age = self._get_memory_age(memory_data)
            
            if age > self.tier_configs['cold']['max_age_hours']:
                # Extract pattern before archiving
                pattern = await self._extract_pattern(memory_data)
                await self._archive_pattern(pattern)
                
                # Remove from cold storage
                del self.memory.tiers['cold'][memory_id]
                migration_counts['cold_to_archive'] += 1
        
        return migration_counts
    
    async def _migrate_memory(
        self,
        memory_id: str,
        from_tier: str,
        to_tier: str
    ):
        """Migrate memory between tiers"""
        
        # Get memory data
        memory_data = self.memory.tiers[from_tier].get(memory_id)
        if not memory_data:
            return
        
        # Apply compression based on target tier
        if self.tier_configs[to_tier]['compression']:
            memory_data = await self._compress_memory(
                memory_data,
                self.tier_configs[to_tier]['compression']
            )
        
        # Update metadata
        memory_data['metadata']['tier'] = to_tier
        memory_data['metadata']['migrated_at'] = datetime.now().isoformat()
        
        # Move to new tier
        self.memory.tiers[to_tier][memory_id] = memory_data
        
        # Remove from old tier
        del self.memory.tiers[from_tier][memory_id]
        
        # Update storage backend
        await self._update_storage_backend(memory_data, to_tier)
    
    async def _compress_memory(
        self,
        memory_data: Dict,
        compression_level: str
    ) -> Dict:
        """Compress memory based on level"""
        
        if compression_level == 'light':
            # Keep essential fields only
            compressed = {
                'content': {
                    'type': memory_data['content'].get('type'),
                    'action': memory_data['content'].get('action'),
                    'result': memory_data['content'].get('success'),
                    'profit': memory_data['content'].get('profit', 0)
                },
                'metadata': memory_data['metadata']
            }
            
        elif compression_level == 'heavy':
            # Keep only critical data
            compressed = {
                'content': {
                    'summary': self._create_summary(memory_data['content'])
                },
                'metadata': {
                    'id': memory_data['metadata']['id'],
                    'timestamp': memory_data['metadata']['timestamp'],
                    'category': memory_data['metadata']['category']
                }
            }
            
        elif compression_level == 'pattern_only':
            # Extract pattern representation
            compressed = {
                'pattern': await self._extract_pattern(memory_data),
                'metadata': {
                    'original_id': memory_data['metadata']['id'],
                    'archived_at': datetime.now().isoformat()
                }
            }
        
        else:
            compressed = memory_data
        
        return compressed
```

## Memory Operations

### Core Operations

```python
class MemoryOperations:
    """Core memory operations"""
    
    async def store_trade_result(
        self,
        action: Dict,
        result: Dict,
        market_context: Dict
    ):
        """Store trade execution result"""
        
        trade_memory = {
            'type': 'trade',
            'action': action['type'],
            'pool': action['pool'],
            'tokens': action['tokens'],
            'amounts': action['amounts'],
            'timestamp': datetime.now(),
            'success': result.get('success', False),
            'tx_hash': result.get('tx_hash'),
            'gas_used': result.get('gas_used', 0),
            'profit': self._calculate_profit(action, result),
            'market_context': market_context,
            'decision_confidence': action.get('confidence', 0)
        }
        
        # Store in memory
        memory_id = await self.add_memory(
            content=trade_memory,
            metadata={
                'category': 'trades',
                'importance': 0.9 if trade_memory['success'] else 0.7
            }
        )
        
        # Check for pattern emergence
        await self._check_pattern_emergence(trade_memory)
        
        return memory_id
    
    async def recall_similar_trades(
        self,
        pool: str,
        action_type: str,
        limit: int = 5
    ) -> List[Dict]:
        """Recall similar past trades"""
        
        query = f"{action_type} pool {pool}"
        
        memories = await self.search_memories(
            query=query,
            limit=limit * 2,
            filters={'category': 'trades'}
        )
        
        # Filter for exact matches
        similar = [
            m for m in memories
            if m['content'].get('pool') == pool and
            m['content'].get('action') == action_type
        ]
        
        # Sort by recency and success
        similar.sort(
            key=lambda m: (
                m['content'].get('success', False),
                m['metadata'].get('timestamp', '')
            ),
            reverse=True
        )
        
        return similar[:limit]
    
    async def get_success_rate(
        self,
        pool: str = None,
        action_type: str = None,
        time_window_days: int = 30
    ) -> float:
        """Calculate success rate for specific conditions"""
        
        # Build query
        query_parts = []
        if pool:
            query_parts.append(f"pool {pool}")
        if action_type:
            query_parts.append(action_type)
        
        query = " ".join(query_parts) if query_parts else "trade"
        
        # Get relevant memories
        memories = await self.search_memories(
            query=query,
            limit=1000,
            filters={'category': 'trades'}
        )
        
        # Filter by time window
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_memories = [
            m for m in memories
            if datetime.fromisoformat(m['metadata']['timestamp']) > cutoff_date
        ]
        
        if not recent_memories:
            return 0.5  # Default to 50% if no data
        
        # Calculate success rate
        successes = sum(1 for m in recent_memories if m['content'].get('success', False))
        
        return successes / len(recent_memories)
```

## Performance Optimization

### Memory Caching

```python
class MemoryCache:
    """High-performance memory cache"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache with LRU update"""
        
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        
        return None
    
    def set(self, key: str, value: Any):
        """Set in cache with LRU eviction"""
        
        # Check if need to evict
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Find least recently used
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries"""
        
        if pattern:
            # Invalidate matching pattern
            keys_to_remove = [k for k in self.cache if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
                del self.access_times[key]
        else:
            # Clear entire cache
            self.cache.clear()
            self.access_times.clear()

class OptimizedMemoryAccess:
    """Optimized memory access patterns"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.cache = MemoryCache()
        self.batch_size = 10
    
    async def batch_search(
        self,
        queries: List[str]
    ) -> Dict[str, List[Dict]]:
        """Batch multiple searches for efficiency"""
        
        results = {}
        
        # Check cache first
        uncached_queries = []
        for query in queries:
            cached = self.cache.get(f"search_{query}")
            if cached:
                results[query] = cached
            else:
                uncached_queries.append(query)
        
        # Batch search uncached queries
        if uncached_queries:
            tasks = [
                self.memory.search_memories(query, limit=10)
                for query in uncached_queries
            ]
            
            search_results = await asyncio.gather(*tasks)
            
            # Cache and return results
            for query, result in zip(uncached_queries, search_results):
                self.cache.set(f"search_{query}", result)
                results[query] = result
        
        return results
    
    async def preload_relevant_memories(
        self,
        context: Dict
    ):
        """Preload memories likely to be needed"""
        
        # Predict what memories will be needed
        predictions = self._predict_memory_needs(context)
        
        # Preload in parallel
        tasks = []
        for prediction in predictions:
            if prediction['type'] == 'search':
                tasks.append(
                    self.memory.search_memories(
                        prediction['query'],
                        limit=prediction.get('limit', 10)
                    )
                )
            elif prediction['type'] == 'pattern':
                tasks.append(
                    self._load_patterns(prediction['pattern_type'])
                )
        
        await asyncio.gather(*tasks)
    
    def _predict_memory_needs(self, context: Dict) -> List[Dict]:
        """Predict what memories will be needed"""
        
        predictions = []
        
        # If analyzing a pool, load its history
        if context.get('pool'):
            predictions.append({
                'type': 'search',
                'query': f"pool {context['pool']}",
                'limit': 20
            })
        
        # If making trade decision, load patterns
        if context.get('decision_type') == 'trade':
            predictions.append({
                'type': 'pattern',
                'pattern_type': 'trade_pattern'
            })
        
        return predictions
```

## Memory Metrics and Monitoring

```python
class MemoryMetrics:
    """Monitor memory system performance"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.metrics = {
            'total_memories': 0,
            'tier_distribution': {},
            'category_distribution': {},
            'pruning_efficiency': 0,
            'pattern_count': 0,
            'compression_ratio': 0,
            'access_patterns': {},
            'storage_usage': {}
        }
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics"""
        
        # Count memories by tier
        for tier_name, tier_data in self.memory.tiers.items():
            self.metrics['tier_distribution'][tier_name] = len(tier_data)
            self.metrics['total_memories'] += len(tier_data)
        
        # Category distribution
        categories = {}
        for tier_data in self.memory.tiers.values():
            for memory in tier_data.values():
                category = memory['metadata'].get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
        
        self.metrics['category_distribution'] = categories
        
        # Pattern count
        self.metrics['pattern_count'] = len(self.memory.patterns)
        
        # Calculate compression ratio
        original_size = self._calculate_original_size()
        current_size = self._calculate_current_size()
        
        if original_size > 0:
            self.metrics['compression_ratio'] = 1 - (current_size / original_size)
        
        # Access patterns
        total_accesses = sum(self.memory.access_counts.values())
        if total_accesses > 0:
            top_accessed = sorted(
                self.memory.access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            self.metrics['access_patterns'] = {
                'total_accesses': total_accesses,
                'top_accessed': top_accessed,
                'avg_accesses': total_accesses / len(self.memory.access_counts)
            }
        
        return self.metrics
```