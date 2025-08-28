"""
High-performance memory caching system with optimization.

This module provides intelligent caching mechanisms to optimize memory
system performance through predictive loading, batch operations, and
smart cache management strategies.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
import hashlib

from .system import MemorySystem


@dataclass
class CacheConfig:
    """Configuration for memory caching"""
    
    max_cache_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour
    lru_threshold: float = 0.8  # Trigger LRU eviction at 80% capacity
    prediction_window_minutes: int = 15
    batch_size: int = 50
    preload_threshold: float = 0.7  # Confidence threshold for preloading
    hit_ratio_target: float = 0.85
    

class CacheEntry:
    """Individual cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.ttl = ttl
        self.size = self._calculate_size(value)
        self.hit_count = 0
        self.prediction_score = 0.0
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of cached value"""
        try:
            if isinstance(value, (dict, list)):
                return len(str(value))
            elif isinstance(value, str):
                return len(value)
            else:
                return 64  # Default size estimate
        except:
            return 64
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        
        return (time.time() - self.created_at) > self.ttl
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1
        self.hit_count += 1
    
    def get_age_seconds(self) -> float:
        """Get age of entry in seconds"""
        return time.time() - self.created_at


class MemoryCache:
    """High-performance LRU cache with TTL and intelligent features"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_times: Dict[str, float] = {}
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expires': 0,
            'preloads': 0,
            'batch_operations': 0
        }
        
        # Access pattern tracking
        self.access_patterns = defaultdict(list)
        self.temporal_patterns = defaultdict(list)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU update"""
        
        entry = self.cache.get(key)
        
        if entry is None:
            self.stats['misses'] += 1
            return None
        
        if entry.is_expired():
            self._remove_entry(key)
            self.stats['expires'] += 1
            self.stats['misses'] += 1
            return None
        
        # Update access statistics
        entry.update_access()
        self.access_times[key] = time.time()
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        # Track access pattern
        self._track_access_pattern(key)
        
        self.stats['hits'] += 1
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with optional TTL"""
        
        # Check if we need to evict
        if len(self.cache) >= self.config.max_cache_size and key not in self.cache:
            self._evict_lru()
        
        # Create new entry
        entry_ttl = ttl or self.config.ttl_seconds
        entry = CacheEntry(key, value, entry_ttl)
        
        self.cache[key] = entry
        self.access_times[key] = time.time()
        
        # Move to end
        self.cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        
        if key in self.cache:
            self._remove_entry(key)
            return True
        
        return False
    
    def _remove_entry(self, key: str):
        """Remove entry and clean up"""
        
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        
        # Clean up access patterns
        if key in self.access_patterns:
            del self.access_patterns[key]
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        
        if not self.cache:
            return
        
        # Get LRU key
        lru_key = next(iter(self.cache))
        self._remove_entry(lru_key)
        self.stats['evictions'] += 1
    
    def _track_access_pattern(self, key: str):
        """Track access patterns for prediction"""
        
        current_time = time.time()
        
        # Track temporal access patterns
        self.access_patterns[key].append(current_time)
        
        # Keep only recent accesses
        cutoff_time = current_time - (self.config.prediction_window_minutes * 60)
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff_time
        ]
        
        # Track hourly patterns
        hour = datetime.fromtimestamp(current_time).hour
        self.temporal_patterns[hour].append(key)
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries"""
        
        if pattern:
            # Invalidate matching pattern
            keys_to_remove = [k for k in self.cache if pattern in k]
            for key in keys_to_remove:
                self._remove_entry(key)
        else:
            # Clear entire cache
            self.cache.clear()
            self.access_times.clear()
            self.access_patterns.clear()
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
            self.stats['expires'] += 1
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_ratio = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'cache_size': len(self.cache),
            'hit_ratio': hit_ratio,
            'total_requests': total_requests,
            'memory_usage': sum(entry.size for entry in self.cache.values())
        }
    
    def predict_needed_keys(self) -> List[str]:
        """Predict keys that might be needed soon"""
        
        predictions = []
        current_time = time.time()
        current_hour = datetime.fromtimestamp(current_time).hour
        
        # Look for patterns based on time
        if current_hour in self.temporal_patterns:
            frequent_keys = set(self.temporal_patterns[current_hour])
            
            # Score by recent frequency
            for key in frequent_keys:
                if key in self.access_patterns:
                    recent_accesses = len([
                        t for t in self.access_patterns[key]
                        if current_time - t < 3600  # Last hour
                    ])
                    
                    if recent_accesses > 0 and key not in self.cache:
                        predictions.append(key)
        
        # Look for sequential patterns
        recently_accessed = [
            key for key, times in self.access_patterns.items()
            if times and current_time - times[-1] < 300  # Last 5 minutes
        ]
        
        for key in recently_accessed:
            # Predict related keys based on naming patterns
            related_keys = self._find_related_keys(key)
            for related_key in related_keys:
                if related_key not in self.cache and related_key not in predictions:
                    predictions.append(related_key)
        
        return predictions[:20]  # Limit predictions
    
    def _find_related_keys(self, key: str) -> List[str]:
        """Find keys that might be related to the given key"""
        
        related = []
        
        # Look for similar patterns in historical data
        key_parts = key.split('_')
        
        for pattern_key in self.access_patterns.keys():
            if key != pattern_key:
                pattern_parts = pattern_key.split('_')
                
                # Check for partial matches
                common_parts = set(key_parts) & set(pattern_parts)
                if len(common_parts) >= len(key_parts) * 0.5:  # 50% similarity
                    related.append(pattern_key)
        
        return related[:5]  # Limit related keys


class OptimizedMemoryAccess:
    """Optimized memory access patterns with intelligent caching"""
    
    def __init__(self, memory_system: MemorySystem, cache_config: CacheConfig = None):
        self.memory = memory_system
        self.cache_config = cache_config or CacheConfig()
        
        # Multiple cache layers
        self.search_cache = MemoryCache(cache_config)
        self.memory_cache = MemoryCache(cache_config)
        self.pattern_cache = MemoryCache(cache_config)
        
        # Batch operation management
        self.pending_operations = []
        self.batch_timer = None
        
        # Prediction and preloading
        self.access_predictor = AccessPredictor(self.memory)
        self.preloader = MemoryPreloader(self.memory, self.memory_cache)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
    
    async def get_memory_cached(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory with caching"""
        
        # Check cache first
        cached_memory = self.memory_cache.get(f"memory_{memory_id}")
        if cached_memory is not None:
            return cached_memory
        
        # Cache miss - fetch from memory system
        memory = await self.memory.operations.get_memory_by_id(memory_id)
        
        if memory:
            # Cache the result
            self.memory_cache.set(f"memory_{memory_id}", memory)
        
        return memory
    
    async def search_memories_cached(
        self,
        query: str,
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search memories with caching"""
        
        # Create cache key
        cache_key = self._create_search_cache_key(query, limit, filters)
        
        # Check cache
        cached_results = self.search_cache.get(cache_key)
        if cached_results is not None:
            return cached_results
        
        # Cache miss - perform search
        results = await self.memory.operations.search_memories(query, limit, filters)
        
        # Cache the results (with shorter TTL for search results)
        self.search_cache.set(cache_key, results, ttl=1800)  # 30 minutes
        
        # Preload related memories
        await self._preload_related_memories(results)
        
        return results
    
    async def batch_search(
        self,
        queries: List[str],
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> Dict[str, List[Dict]]:
        """Batch multiple searches for efficiency"""
        
        start_time = time.time()
        results = {}
        
        # Check cache first
        uncached_queries = []
        for query in queries:
            cache_key = self._create_search_cache_key(query, limit, filters)
            cached = self.search_cache.get(cache_key)
            
            if cached:
                results[query] = cached
            else:
                uncached_queries.append(query)
        
        # Batch search uncached queries
        if uncached_queries:
            # Execute searches in parallel
            search_tasks = [
                self.memory.operations.search_memories(query, limit, filters)
                for query in uncached_queries
            ]
            
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Cache and store results
            for query, result in zip(uncached_queries, search_results):
                if not isinstance(result, Exception):
                    cache_key = self._create_search_cache_key(query, limit, filters)
                    self.search_cache.set(cache_key, result, ttl=1800)
                    results[query] = result
                else:
                    results[query] = []
        
        # Track performance
        duration = time.time() - start_time
        self.performance_tracker.track_batch_operation(len(queries), duration)
        
        # Update cache stats
        self.search_cache.stats['batch_operations'] += 1
        
        return results
    
    async def preload_relevant_memories(self, context: Dict[str, Any]):
        """Preload memories likely to be needed based on context"""
        
        predictions = await self.access_predictor.predict_needed_memories(context)
        
        if predictions:
            await self.preloader.preload_memories(predictions)
    
    async def _preload_related_memories(self, search_results: List[Dict]):
        """Preload memories related to search results"""
        
        related_ids = []
        
        for result in search_results[:5]:  # Only top 5 results
            memory_id = result.get('metadata', {}).get('id')
            if memory_id:
                # Find related memories based on content
                related = await self._find_related_memory_ids(result)
                related_ids.extend(related[:3])  # Limit to 3 per result
        
        if related_ids:
            await self.preloader.preload_memory_ids(related_ids)
    
    async def _find_related_memory_ids(self, memory: Dict) -> List[str]:
        """Find IDs of memories related to given memory"""
        
        related_ids = []
        content = memory.get('content', {})
        
        # Look for memories with same pool
        pool = content.get('pool')
        if pool:
            pool_memories = await self.search_memories_cached(
                query=f"pool {pool}",
                limit=5
            )
            
            for mem in pool_memories:
                mem_id = mem.get('metadata', {}).get('id')
                if mem_id and mem_id != memory.get('metadata', {}).get('id'):
                    related_ids.append(mem_id)
        
        # Look for memories with same action type
        action = content.get('action')
        if action:
            action_memories = await self.search_memories_cached(
                query=f"action {action}",
                limit=3
            )
            
            for mem in action_memories:
                mem_id = mem.get('metadata', {}).get('id')
                if mem_id and mem_id not in related_ids:
                    related_ids.append(mem_id)
        
        return related_ids[:5]  # Limit total related memories
    
    def _create_search_cache_key(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Create cache key for search"""
        
        # Create deterministic key
        key_parts = [query, str(limit)]
        
        if filters:
            # Sort filters for consistent key
            sorted_filters = sorted(filters.items())
            key_parts.append(str(sorted_filters))
        
        key_string = "|".join(key_parts)
        
        # Hash for consistent length
        return f"search_{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def invalidate_related_caches(self, memory_id: str, content: Dict):
        """Invalidate caches related to a memory change"""
        
        # Invalidate direct memory cache
        self.memory_cache.delete(f"memory_{memory_id}")
        
        # Invalidate search caches that might contain this memory
        pool = content.get('pool')
        if pool:
            self.search_cache.invalidate(f"pool_{pool}")
        
        action = content.get('action')
        if action:
            self.search_cache.invalidate(f"action_{action}")
        
        category = content.get('category')
        if category:
            self.search_cache.invalidate(f"category_{category}")
    
    async def optimize_cache_performance(self):
        """Optimize cache performance"""
        
        # Clean up expired entries
        expired_search = self.search_cache.cleanup_expired()
        expired_memory = self.memory_cache.cleanup_expired()
        expired_pattern = self.pattern_cache.cleanup_expired()
        
        # Analyze hit ratios
        search_stats = self.search_cache.get_stats()
        memory_stats = self.memory_cache.get_stats()
        
        # Adjust cache sizes if needed
        if search_stats['hit_ratio'] < self.cache_config.hit_ratio_target:
            # Consider increasing search cache size
            self.cache_config.max_cache_size = min(
                self.cache_config.max_cache_size * 1.2,
                2000  # Cap at 2000
            )
        
        # Preload based on predictions
        predictions = self.search_cache.predict_needed_keys()
        if predictions:
            await self._preload_predicted_searches(predictions)
        
        return {
            'expired_entries': expired_search + expired_memory + expired_pattern,
            'search_hit_ratio': search_stats['hit_ratio'],
            'memory_hit_ratio': memory_stats['hit_ratio'],
            'predictions_made': len(predictions)
        }
    
    async def _preload_predicted_searches(self, predicted_keys: List[str]):
        """Preload searches based on predictions"""
        
        for cache_key in predicted_keys[:5]:  # Limit to avoid overload
            # Try to extract query from cache key
            if cache_key.startswith('search_'):
                # This is a simplified approach - in practice you'd need
                # more sophisticated key parsing
                try:
                    # Placeholder for search preloading
                    # In practice, you'd decode the cache key and execute the search
                    pass
                except Exception as e:
                    continue
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        return {
            'search_cache': self.search_cache.get_stats(),
            'memory_cache': self.memory_cache.get_stats(),
            'pattern_cache': self.pattern_cache.get_stats(),
            'performance': self.performance_tracker.get_stats(),
            'cache_config': {
                'max_size': self.cache_config.max_cache_size,
                'ttl_seconds': self.cache_config.ttl_seconds,
                'hit_ratio_target': self.cache_config.hit_ratio_target
            }
        }


class AccessPredictor:
    """Predict which memories will be accessed based on patterns"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.access_history = defaultdict(list)
        self.pattern_weights = {
            'temporal': 0.3,
            'content_similarity': 0.4,
            'sequential': 0.3
        }
    
    async def predict_needed_memories(
        self,
        context: Dict[str, Any],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Predict memories that might be needed"""
        
        predictions = []
        
        # Temporal predictions
        temporal_predictions = await self._predict_by_time_patterns()
        predictions.extend(temporal_predictions)
        
        # Content-based predictions
        if context:
            content_predictions = await self._predict_by_content_similarity(context)
            predictions.extend(content_predictions)
        
        # Sequential predictions
        sequential_predictions = await self._predict_by_sequence_patterns()
        predictions.extend(sequential_predictions)
        
        # Remove duplicates and sort by prediction confidence
        unique_predictions = {}
        for pred in predictions:
            memory_id = pred.get('memory_id')
            if memory_id not in unique_predictions:
                unique_predictions[memory_id] = pred
            else:
                # Combine confidence scores
                existing = unique_predictions[memory_id]
                existing['confidence'] = max(existing['confidence'], pred['confidence'])
        
        # Sort by confidence and return top predictions
        sorted_predictions = sorted(
            unique_predictions.values(),
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return sorted_predictions[:limit]
    
    async def _predict_by_time_patterns(self) -> List[Dict[str, Any]]:
        """Predict based on temporal access patterns"""
        
        predictions = []
        current_hour = datetime.now().hour
        
        # Look for memories commonly accessed at this time
        # This would analyze historical access patterns
        # Placeholder implementation
        
        return predictions
    
    async def _predict_by_content_similarity(
        self,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Predict based on content similarity to current context"""
        
        predictions = []
        
        # Extract key context elements
        pool = context.get('pool')
        action = context.get('action_type')
        
        if pool:
            # Predict memories related to this pool
            pool_query = f"pool {pool}"
            similar_memories = await self.memory.operations.search_memories(
                query=pool_query,
                limit=10
            )
            
            for memory in similar_memories:
                memory_id = memory.get('metadata', {}).get('id')
                if memory_id:
                    predictions.append({
                        'memory_id': memory_id,
                        'confidence': 0.7,
                        'reason': f'pool_similarity_{pool}'
                    })
        
        if action:
            # Predict memories related to this action
            action_query = f"action {action}"
            similar_memories = await self.memory.operations.search_memories(
                query=action_query,
                limit=5
            )
            
            for memory in similar_memories:
                memory_id = memory.get('metadata', {}).get('id')
                if memory_id:
                    predictions.append({
                        'memory_id': memory_id,
                        'confidence': 0.6,
                        'reason': f'action_similarity_{action}'
                    })
        
        return predictions
    
    async def _predict_by_sequence_patterns(self) -> List[Dict[str, Any]]:
        """Predict based on sequential access patterns"""
        
        predictions = []
        
        # Analyze recent access sequences to predict next likely accesses
        # This would look at the last few accessed memories and predict
        # what might come next based on historical patterns
        # Placeholder implementation
        
        return predictions


class MemoryPreloader:
    """Preload memories into cache based on predictions"""
    
    def __init__(self, memory_system: MemorySystem, cache: MemoryCache):
        self.memory = memory_system
        self.cache = cache
        self.preload_stats = {
            'preloads_attempted': 0,
            'preloads_successful': 0,
            'preloads_failed': 0,
            'cache_hits_from_preload': 0
        }
    
    async def preload_memories(self, predictions: List[Dict[str, Any]]):
        """Preload memories based on predictions"""
        
        # Filter by confidence threshold
        high_confidence = [
            p for p in predictions 
            if p.get('confidence', 0) >= 0.7
        ]
        
        # Limit concurrent preloads
        semaphore = asyncio.Semaphore(5)
        
        async def preload_single(prediction):
            async with semaphore:
                await self._preload_single_memory(prediction)
        
        # Execute preloads
        tasks = [preload_single(pred) for pred in high_confidence[:10]]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def preload_memory_ids(self, memory_ids: List[str]):
        """Preload specific memory IDs"""
        
        for memory_id in memory_ids:
            if not self.cache.get(f"memory_{memory_id}"):
                await self._preload_memory_by_id(memory_id)
    
    async def _preload_single_memory(self, prediction: Dict[str, Any]):
        """Preload a single memory"""
        
        memory_id = prediction.get('memory_id')
        if not memory_id:
            return
        
        self.preload_stats['preloads_attempted'] += 1
        
        try:
            # Check if already cached
            cache_key = f"memory_{memory_id}"
            if self.cache.get(cache_key):
                return  # Already cached
            
            # Fetch and cache
            memory = await self.memory.operations.get_memory_by_id(memory_id)
            if memory:
                self.cache.set(cache_key, memory)
                self.preload_stats['preloads_successful'] += 1
            else:
                self.preload_stats['preloads_failed'] += 1
        
        except Exception as e:
            self.preload_stats['preloads_failed'] += 1
    
    async def _preload_memory_by_id(self, memory_id: str):
        """Preload memory by ID"""
        
        try:
            memory = await self.memory.operations.get_memory_by_id(memory_id)
            if memory:
                cache_key = f"memory_{memory_id}"
                self.cache.set(cache_key, memory)
                self.preload_stats['preloads_successful'] += 1
        except Exception as e:
            self.preload_stats['preloads_failed'] += 1
    
    def get_preload_stats(self) -> Dict[str, Any]:
        """Get preloading statistics"""
        
        total_attempts = self.preload_stats['preloads_attempted']
        success_rate = (
            self.preload_stats['preloads_successful'] / total_attempts
            if total_attempts > 0 else 0
        )
        
        return {
            **self.preload_stats,
            'success_rate': success_rate
        }


class PerformanceTracker:
    """Track cache performance metrics"""
    
    def __init__(self):
        self.operation_times = []
        self.batch_operations = []
        self.cache_effectiveness = []
    
    def track_operation(self, operation_type: str, duration: float):
        """Track individual operation"""
        
        self.operation_times.append({
            'type': operation_type,
            'duration': duration,
            'timestamp': time.time()
        })
        
        # Keep only recent operations
        cutoff_time = time.time() - 3600  # 1 hour
        self.operation_times = [
            op for op in self.operation_times
            if op['timestamp'] > cutoff_time
        ]
    
    def track_batch_operation(self, batch_size: int, duration: float):
        """Track batch operation performance"""
        
        self.batch_operations.append({
            'size': batch_size,
            'duration': duration,
            'per_item': duration / batch_size if batch_size > 0 else 0,
            'timestamp': time.time()
        })
        
        # Keep only recent batches
        cutoff_time = time.time() - 3600
        self.batch_operations = [
            batch for batch in self.batch_operations
            if batch['timestamp'] > cutoff_time
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        if not self.operation_times and not self.batch_operations:
            return {
                'avg_operation_time': 0,
                'avg_batch_time': 0,
                'operation_count': 0,
                'batch_count': 0
            }
        
        # Calculate averages
        avg_op_time = (
            sum(op['duration'] for op in self.operation_times) / len(self.operation_times)
            if self.operation_times else 0
        )
        
        avg_batch_time = (
            sum(batch['duration'] for batch in self.batch_operations) / len(self.batch_operations)
            if self.batch_operations else 0
        )
        
        return {
            'avg_operation_time': avg_op_time,
            'avg_batch_time': avg_batch_time,
            'operation_count': len(self.operation_times),
            'batch_count': len(self.batch_operations),
            'recent_operations': self.operation_times[-10:],  # Last 10
            'recent_batches': self.batch_operations[-5:]  # Last 5
        }