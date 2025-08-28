"""
Core memory system with Mem0 integration.

This module provides the main MemorySystem class that orchestrates all
memory operations including storage, retrieval, and management.
"""

import os
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from mem0 import Memory


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
    
    # Performance settings
    enable_caching: bool = True
    enable_monitoring: bool = True
    enable_auto_pruning: bool = True
    pruning_interval_hours: int = 24
    
    # Mem0 configuration
    openai_api_key: Optional[str] = None
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    history_db_path: str = "./memory_history.db"
    
    def __post_init__(self):
        """Set default API key from environment"""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")


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
        
        # Component initialization flags
        self._initialized_components = False
    
    def _initialize_mem0(self) -> Memory:
        """Initialize Mem0 with configuration"""
        
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key is required for memory system")
        
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": self.config.openai_api_key,
                    "model": "gpt-4-turbo",
                    "temperature": 0.1  # Low temperature for consistency
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": self.config.openai_api_key,
                    "model": "text-embedding-3-small"
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "aerodrome_brain",
                    "host": self.config.qdrant_host,
                    "port": self.config.qdrant_port,
                    "embedding_model_dims": 1536
                }
            },
            "history_db_path": self.config.history_db_path,
            "version": "v1.1"
        }
        
        return Memory.from_config(config)
    
    async def initialize_components(self):
        """Initialize memory system components"""
        if self._initialized_components:
            return
        
        print("Initializing memory system components...")
        
        # Import components to avoid circular imports
        from .operations import MemoryOperations
        from .categories import MemoryCategories
        from .pruning import MemoryPruning
        from .patterns import PatternExtractor
        from .tiers import StorageTiers
        from .cache import OptimizedMemoryAccess, CacheConfig
        from .metrics import MemoryMetrics
        
        # Initialize components
        print("Initializing operations...")
        self.operations = MemoryOperations(self)
        
        print("Initializing categories...")
        self.categories = MemoryCategories()
        
        print("Initializing pruning...")
        self.pruning = MemoryPruning(self)
        
        print("Initializing pattern extractor...")
        self.pattern_extractor = PatternExtractor(self)
        
        print("Initializing tiers manager...")
        self.tiers_manager = StorageTiers(self)
        
        # Initialize caching if enabled
        if self.config.enable_caching:
            print("Initializing optimized access with caching...")
            cache_config = CacheConfig(
                max_cache_size=self.config.max_hot_memories,
                ttl_seconds=3600,  # 1 hour
                hit_ratio_target=0.8
            )
            self.optimized_access = OptimizedMemoryAccess(self, cache_config)
        
        # Initialize monitoring if enabled
        if self.config.enable_monitoring:
            print("Initializing metrics monitoring...")
            self.metrics = MemoryMetrics(self)
            
            # Start monitoring in background
            await self.metrics.start_monitoring()
        
        # Set up auto-pruning if enabled
        if self.config.enable_auto_pruning:
            print("Setting up auto-pruning...")
            self._setup_auto_pruning()
        
        print("Memory system components initialized successfully")
        self._initialized_components = True
    
    async def add_memory(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add memory to the system"""
        await self.initialize_components()
        return await self.operations.add_memory(content, metadata)
    
    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search memories using semantic search"""
        await self.initialize_components()
        return await self.operations.search_memories(query, limit, filters)
    
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update existing memory"""
        await self.initialize_components()
        return await self.operations.update_memory(memory_id, updates)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from the system"""
        await self.initialize_components()
        return await self.operations.delete_memory(memory_id)
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory by ID"""
        await self.initialize_components()
        return await self.operations.get_memory_by_id(memory_id)
    
    async def store_trade_result(
        self,
        action: Dict,
        result: Dict,
        market_context: Dict
    ) -> str:
        """Store trade execution result"""
        await self.initialize_components()
        return await self.operations.store_trade_result(action, result, market_context)
    
    async def recall_similar_trades(
        self,
        pool: str,
        action_type: str,
        limit: int = 5
    ) -> List[Dict]:
        """Recall similar past trades"""
        await self.initialize_components()
        return await self.operations.recall_similar_trades(pool, action_type, limit)
    
    async def get_success_rate(
        self,
        pool: str = None,
        action_type: str = None,
        time_window_days: int = 30
    ) -> float:
        """Calculate success rate for specific conditions"""
        await self.initialize_components()
        return await self.operations.get_success_rate(pool, action_type, time_window_days)
    
    async def execute_pruning_cycle(self) -> Dict[str, Any]:
        """Execute complete pruning cycle"""
        await self.initialize_components()
        return await self.pruning.execute_pruning_cycle()
    
    async def extract_patterns(self) -> List[Dict]:
        """Extract patterns from memory history"""
        await self.initialize_components()
        return await self.pattern_extractor.extract_patterns()
    
    async def migrate_tiers(self) -> Dict[str, int]:
        """Migrate memories between tiers"""
        await self.initialize_components()
        return await self.tiers_manager.migrate_tiers()
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics"""
        await self.initialize_components()
        return await self.metrics.collect_metrics()
    
    def generate_memory_id(self, content: Dict) -> str:
        """Generate unique memory ID"""
        # Create hash from content
        content_str = json.dumps(content, sort_keys=True)
        hash_obj = hashlib.sha256(content_str.encode())
        
        # Add timestamp for uniqueness
        timestamp = int(datetime.now().timestamp())
        
        return f"mem_{hash_obj.hexdigest()[:12]}_{timestamp}"
    
    def format_memory(self, content: Dict) -> str:
        """Format memory content for storage"""
        memory_type = content.get('type', 'general')
        
        if memory_type == 'trade':
            return self._format_trade_memory(content)
        elif memory_type == 'market_observation':
            return self._format_market_memory(content)
        elif memory_type == 'pattern':
            return self._format_pattern_memory(content)
        else:
            return json.dumps(content, indent=2)
    
    def _format_trade_memory(self, content: Dict) -> str:
        """Format trade memory"""
        return (
            f"Trade: {content.get('action', 'unknown')} "
            f"{content.get('amount', 'N/A')} {content.get('token', 'N/A')} "
            f"in pool {content.get('pool', 'N/A')}. "
            f"Result: {content.get('result', 'pending')}. "
            f"Profit: {content.get('profit', 0)}. "
            f"Gas: {content.get('gas_used', 0)}."
        )
    
    def _format_market_memory(self, content: Dict) -> str:
        """Format market observation memory"""
        return (
            f"Market observation: {content.get('observation', 'N/A')} "
            f"for pool {content.get('pool', 'N/A')} "
            f"at price {content.get('price', 'N/A')}. "
            f"Volume: {content.get('volume', 'N/A')}. "
            f"Conditions: {content.get('conditions', {})}"
        )
    
    def _format_pattern_memory(self, content: Dict) -> str:
        """Format pattern memory"""
        return (
            f"Pattern: {content.get('pattern_type', 'unknown')} "
            f"with {content.get('occurrence_count', 0)} occurrences. "
            f"Success rate: {content.get('success_rate', 0):.2%}. "
            f"Confidence: {content.get('confidence', 0):.2f}. "
            f"Conditions: {content.get('conditions', {})}"
        )
    
    def categorize_memory(self, content: Dict) -> str:
        """Categorize memory based on content"""
        if not self._initialized_components:
            # Simple categorization without full initialization
            memory_type = content.get('type', '')
            
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
            
            return 'market_observations'
        
        return self.categories.categorize_memory(content)
    
    def calculate_ttl(self, content: Dict) -> int:
        """Calculate time-to-live for memory"""
        category = self.categorize_memory(content)
        
        # Default TTL values by category (in days)
        ttl_mapping = {
            'trades': 30,
            'market_observations': 7,
            'patterns': 365,
            'failures': 60,
            'opportunities': 3,
            'user_preferences': -1,  # Never expire
        }
        
        return ttl_mapping.get(category, 30)
    
    def get_tier_for_memory(self, memory_data: Dict) -> str:
        """Determine appropriate tier for memory"""
        metadata = memory_data.get('metadata', {})
        
        # Check if already assigned
        if 'tier' in metadata:
            return metadata['tier']
        
        # Calculate age
        timestamp_str = metadata.get('timestamp', datetime.now().isoformat())
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        
        # Assign tier based on age
        if age_hours <= self.config.hot_threshold:
            return 'hot'
        elif age_hours <= self.config.warm_threshold:
            return 'warm'
        elif age_hours <= self.config.cold_threshold:
            return 'cold'
        else:
            return 'archive'
    
    async def backup_memory_state(self, backup_path: str):
        """Backup current memory state"""
        backup_data = {
            'config': asdict(self.config),
            'tiers': self.tiers,
            'patterns': self.patterns,
            'access_counts': self.access_counts,
            'last_access': {k: v.isoformat() for k, v in self.last_access.items()},
            'backup_timestamp': datetime.now().isoformat()
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
    
    async def restore_memory_state(self, backup_path: str):
        """Restore memory state from backup"""
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        
        # Restore tiers
        self.tiers = backup_data.get('tiers', {})
        self.patterns = backup_data.get('patterns', {})
        self.access_counts = backup_data.get('access_counts', {})
        
        # Restore last access times
        last_access_data = backup_data.get('last_access', {})
        self.last_access = {
            k: datetime.fromisoformat(v) for k, v in last_access_data.items()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on memory system"""
        try:
            # Test Mem0 connection
            test_memory = {
                'type': 'health_check',
                'content': 'System health check',
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to add and immediately delete a test memory
            test_id = self.generate_memory_id(test_memory)
            
            # Basic functionality test
            await self.add_memory(test_memory)
            results = await self.search_memories("health check", limit=1)
            
            # Clean up test memory
            if results:
                await self.delete_memory(test_id)
            
            return {
                'status': 'healthy',
                'mem0_connected': True,
                'components_initialized': self._initialized_components,
                'total_memories': sum(len(tier) for tier in self.tiers.values()),
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'mem0_connected': False,
                'components_initialized': self._initialized_components,
                'last_check': datetime.now().isoformat()
            }
    
    def _setup_auto_pruning(self):
        """Set up automatic pruning"""
        if not hasattr(self, '_pruning_task') or self._pruning_task is None:
            self._pruning_task = asyncio.create_task(self._auto_pruning_loop())
    
    async def _auto_pruning_loop(self):
        """Background auto-pruning loop"""
        try:
            while self.config.enable_auto_pruning:
                # Wait for pruning interval
                await asyncio.sleep(self.config.pruning_interval_hours * 3600)
                
                # Execute pruning cycle
                try:
                    print("Starting automatic pruning cycle...")
                    result = await self.execute_pruning_cycle()
                    print(f"Auto-pruning completed: {result}")
                except Exception as e:
                    print(f"Auto-pruning failed: {str(e)}")
                
        except asyncio.CancelledError:
            print("Auto-pruning loop cancelled")
        except Exception as e:
            print(f"Auto-pruning loop error: {str(e)}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        await self.initialize_components()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'initialized': self._initialized_components,
            'config': asdict(self.config),
            'components': {
                'operations': hasattr(self, 'operations'),
                'categories': hasattr(self, 'categories'),
                'pruning': hasattr(self, 'pruning'),
                'pattern_extractor': hasattr(self, 'pattern_extractor'),
                'tiers_manager': hasattr(self, 'tiers_manager'),
                'optimized_access': hasattr(self, 'optimized_access'),
                'metrics': hasattr(self, 'metrics')
            }
        }
        
        # Add tier statistics
        if hasattr(self, 'tiers_manager'):
            try:
                tier_stats = await self.tiers_manager.get_tier_statistics()
                status['tier_statistics'] = tier_stats
            except Exception as e:
                status['tier_statistics_error'] = str(e)
        
        # Add metrics summary
        if hasattr(self, 'metrics'):
            try:
                metrics_summary = self.metrics.get_current_metrics_summary()
                status['metrics_summary'] = metrics_summary
            except Exception as e:
                status['metrics_error'] = str(e)
        
        # Add cache statistics
        if hasattr(self, 'optimized_access'):
            try:
                cache_stats = self.optimized_access.get_cache_stats()
                status['cache_statistics'] = cache_stats
            except Exception as e:
                status['cache_error'] = str(e)
        
        # Add pattern summary
        if hasattr(self, 'pattern_extractor'):
            try:
                pattern_summary = self.pattern_extractor.get_pattern_summary()
                status['pattern_summary'] = pattern_summary
            except Exception as e:
                status['pattern_error'] = str(e)
        
        return status
    
    async def optimize_system_performance(self) -> Dict[str, Any]:
        """Optimize overall system performance"""
        await self.initialize_components()
        
        optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'optimizations_performed': []
        }
        
        # Optimize cache performance
        if hasattr(self, 'optimized_access'):
            try:
                cache_results = await self.optimized_access.optimize_cache_performance()
                optimization_results['cache_optimization'] = cache_results
                optimization_results['optimizations_performed'].append('cache')
            except Exception as e:
                optimization_results['cache_optimization_error'] = str(e)
        
        # Optimize tier performance
        if hasattr(self, 'tiers_manager'):
            try:
                tier_results = await self.tiers_manager.optimize_tier_performance()
                optimization_results['tier_optimization'] = tier_results
                optimization_results['optimizations_performed'].append('tiers')
            except Exception as e:
                optimization_results['tier_optimization_error'] = str(e)
        
        # Clean up old metrics data
        if hasattr(self, 'metrics'):
            try:
                await self.metrics.cleanup_old_data()
                optimization_results['optimizations_performed'].append('metrics_cleanup')
            except Exception as e:
                optimization_results['metrics_cleanup_error'] = str(e)
        
        # Extract fresh patterns
        try:
            patterns = await self.extract_patterns()
            optimization_results['new_patterns_extracted'] = len(patterns)
            optimization_results['optimizations_performed'].append('pattern_extraction')
        except Exception as e:
            optimization_results['pattern_extraction_error'] = str(e)
        
        return optimization_results
    
    async def search_with_caching(
        self,
        query: str,
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search memories with caching if available"""
        await self.initialize_components()
        
        # Use optimized access if available
        if hasattr(self, 'optimized_access'):
            return await self.optimized_access.search_memories_cached(query, limit, filters)
        else:
            # Fallback to standard search
            return await self.search_memories(query, limit, filters)
    
    async def get_memory_with_caching(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory by ID with caching if available"""
        await self.initialize_components()
        
        # Use optimized access if available
        if hasattr(self, 'optimized_access'):
            return await self.optimized_access.get_memory_cached(memory_id)
        else:
            # Fallback to standard retrieval
            return await self.get_memory_by_id(memory_id)
    
    async def batch_search(
        self,
        queries: List[str],
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> Dict[str, List[Dict]]:
        """Perform batch searches for efficiency"""
        await self.initialize_components()
        
        # Use optimized access if available
        if hasattr(self, 'optimized_access'):
            return await self.optimized_access.batch_search(queries, limit, filters)
        else:
            # Fallback to sequential searches
            results = {}
            for query in queries:
                results[query] = await self.search_memories(query, limit, filters)
            return results
    
    async def preload_context_memories(self, context: Dict[str, Any]):
        """Preload memories based on context"""
        await self.initialize_components()
        
        if hasattr(self, 'optimized_access'):
            await self.optimized_access.preload_relevant_memories(context)
    
    async def export_system_data(self, export_path: str, include_metrics: bool = True):
        """Export comprehensive system data"""
        await self.initialize_components()
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_status': await self.get_system_status(),
            'memory_backup': {
                'tiers': self.tiers,
                'patterns': self.patterns,
                'access_counts': self.access_counts,
                'last_access': {k: v.isoformat() for k, v in self.last_access.items()}
            }
        }
        
        # Export metrics if requested and available
        if include_metrics and hasattr(self, 'metrics'):
            try:
                metrics_file = export_path.replace('.json', '_metrics.json')
                await self.metrics.export_metrics(metrics_file)
                export_data['metrics_exported_to'] = metrics_file
            except Exception as e:
                export_data['metrics_export_error'] = str(e)
        
        # Save main export data
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"System data exported to {export_path}")
    
    async def shutdown(self):
        """Gracefully shutdown memory system"""
        print("Starting memory system shutdown...")
        
        # Stop monitoring
        if hasattr(self, 'metrics'):
            try:
                await self.metrics.stop_monitoring()
            except Exception as e:
                print(f"Error stopping monitoring: {str(e)}")
        
        # Stop auto-pruning
        if hasattr(self, '_pruning_task') and self._pruning_task:
            self._pruning_task.cancel()
            try:
                await self._pruning_task
            except asyncio.CancelledError:
                pass
        
        # Perform final pruning cycle
        if self._initialized_components and hasattr(self, 'pruning'):
            try:
                print("Performing final pruning cycle...")
                await self.execute_pruning_cycle()
            except Exception as e:
                print(f"Final pruning failed: {str(e)}")
        
        # Optimize system one last time
        try:
            print("Final system optimization...")
            await self.optimize_system_performance()
        except Exception as e:
            print(f"Final optimization failed: {str(e)}")
        
        # Clear caches
        if hasattr(self, 'optimized_access'):
            try:
                self.optimized_access.search_cache.invalidate()
                self.optimized_access.memory_cache.invalidate()
                self.optimized_access.pattern_cache.invalidate()
            except Exception as e:
                print(f"Error clearing caches: {str(e)}")
        
        # Close Mem0 connections if possible
        # Note: Mem0 doesn't expose explicit close methods
        
        print(f"Memory system shutdown completed at {datetime.now().isoformat()}")