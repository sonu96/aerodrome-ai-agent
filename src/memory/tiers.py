"""
Multi-tier storage management for the memory system.

This module manages the hierarchical storage system with hot, warm, cold,
and archive tiers, implementing automatic migration and compression strategies.
"""

import json
import asyncio
import gzip
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from .system import MemorySystem


@dataclass
class TierConfiguration:
    """Configuration for a storage tier"""
    
    name: str
    max_age_hours: int  # -1 for infinite
    max_size: int  # -1 for unlimited
    compression_level: str  # 'none', 'light', 'heavy', 'pattern_only'
    storage_backend: str  # 'memory', 'disk', 'cloud'
    auto_migration: bool = True
    access_pattern: str = 'lru'  # 'lru', 'lfu', 'fifo'


class StorageTiers:
    """Manage multi-tier storage system"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.tier_configs = self._initialize_tier_configs()
        self.migration_stats = {
            'total_migrations': 0,
            'hot_to_warm': 0,
            'warm_to_cold': 0,
            'cold_to_archive': 0,
            'last_migration': None
        }
        
        # Storage backends
        self.storage_backends = {
            'memory': MemoryBackend(),
            'disk': DiskBackend(base_path='./memory_storage'),
            'cloud': CloudBackend()  # Placeholder for cloud storage
        }
        
        # Ensure storage directories exist
        self._ensure_storage_directories()
    
    def _initialize_tier_configs(self) -> Dict[str, TierConfiguration]:
        """Initialize tier configurations"""
        
        return {
            'hot': TierConfiguration(
                name='hot',
                max_age_hours=24,
                max_size=1000,
                compression_level='none',
                storage_backend='memory',
                auto_migration=True,
                access_pattern='lru'
            ),
            
            'warm': TierConfiguration(
                name='warm',
                max_age_hours=168,  # 7 days
                max_size=5000,
                compression_level='light',
                storage_backend='disk',
                auto_migration=True,
                access_pattern='lru'
            ),
            
            'cold': TierConfiguration(
                name='cold',
                max_age_hours=720,  # 30 days
                max_size=10000,
                compression_level='heavy',
                storage_backend='disk',
                auto_migration=True,
                access_pattern='lfu'
            ),
            
            'archive': TierConfiguration(
                name='archive',
                max_age_hours=-1,  # Infinite
                max_size=-1,  # Unlimited
                compression_level='pattern_only',
                storage_backend='cloud',
                auto_migration=False,
                access_pattern='fifo'
            )
        }
    
    def _ensure_storage_directories(self):
        """Ensure storage directories exist"""
        
        base_path = Path('./memory_storage')
        for tier_name in ['warm', 'cold', 'archive']:
            tier_path = base_path / tier_name
            tier_path.mkdir(parents=True, exist_ok=True)
    
    async def migrate_tiers(self) -> Dict[str, int]:
        """Migrate memories between tiers based on age and access patterns"""
        
        migration_counts = {
            'hot_to_warm': 0,
            'warm_to_cold': 0,
            'cold_to_archive': 0,
            'errors': 0
        }
        
        print("Starting tier migration...")
        
        try:
            # Hot to Warm migration
            hot_migrations = await self._migrate_tier('hot', 'warm')
            migration_counts['hot_to_warm'] = hot_migrations
            
            # Warm to Cold migration
            warm_migrations = await self._migrate_tier('warm', 'cold')
            migration_counts['warm_to_cold'] = warm_migrations
            
            # Cold to Archive migration
            cold_migrations = await self._migrate_tier('cold', 'archive')
            migration_counts['cold_to_archive'] = cold_migrations
            
            # Update stats
            total_migrations = sum([
                migration_counts['hot_to_warm'],
                migration_counts['warm_to_cold'],
                migration_counts['cold_to_archive']
            ])
            
            self.migration_stats.update({
                'total_migrations': self.migration_stats['total_migrations'] + total_migrations,
                'hot_to_warm': self.migration_stats['hot_to_warm'] + migration_counts['hot_to_warm'],
                'warm_to_cold': self.migration_stats['warm_to_cold'] + migration_counts['warm_to_cold'],
                'cold_to_archive': self.migration_stats['cold_to_archive'] + migration_counts['cold_to_archive'],
                'last_migration': datetime.now().isoformat()
            })
            
            print(f"Migration completed: {migration_counts}")
            
        except Exception as e:
            print(f"Error during tier migration: {str(e)}")
            migration_counts['errors'] += 1
        
        return migration_counts
    
    async def _migrate_tier(self, from_tier: str, to_tier: str) -> int:
        """Migrate memories from one tier to another"""
        
        migration_count = 0
        
        from_config = self.tier_configs[from_tier]
        to_config = self.tier_configs[to_tier]
        
        # Get memories to migrate
        candidates = await self._get_migration_candidates(from_tier, to_tier)
        
        for memory_id, memory_data in candidates:
            try:
                # Apply compression if needed
                if to_config.compression_level != 'none':
                    memory_data = await self._compress_memory(memory_data, to_config.compression_level)
                
                # Update metadata
                memory_data['metadata']['tier'] = to_tier
                memory_data['metadata']['migrated_at'] = datetime.now().isoformat()
                memory_data['metadata']['compression_applied'] = to_config.compression_level
                
                # Handle special archive migration
                if to_tier == 'archive':
                    # Extract pattern before archiving
                    pattern = await self._extract_pattern_for_archive(memory_data)
                    if pattern:
                        await self._archive_pattern(pattern)
                    
                    # Remove from source tier
                    del self.memory.tiers[from_tier][memory_id]
                    migration_count += 1
                    continue
                
                # Move to new tier
                self.memory.tiers[to_tier][memory_id] = memory_data
                
                # Update storage backend
                await self._update_storage_backend(memory_data, from_config, to_config)
                
                # Remove from old tier
                del self.memory.tiers[from_tier][memory_id]
                
                migration_count += 1
                
            except Exception as e:
                print(f"Error migrating memory {memory_id} from {from_tier} to {to_tier}: {str(e)}")
                continue
        
        return migration_count
    
    async def _get_migration_candidates(self, from_tier: str, to_tier: str) -> List[Tuple[str, Dict]]:
        """Get memories eligible for migration"""
        
        candidates = []
        current_time = datetime.now()
        
        from_config = self.tier_configs[from_tier]
        tier_memories = self.memory.tiers[from_tier]
        
        for memory_id, memory_data in tier_memories.items():
            try:
                metadata = memory_data.get('metadata', {})
                
                # Calculate memory age
                age_hours = self._get_memory_age_hours(metadata)
                if age_hours is None:
                    continue
                
                # Check if memory should migrate based on age
                if age_hours > from_config.max_age_hours:
                    # Additional checks for migration eligibility
                    if await self._is_migration_eligible(memory_data, from_tier, to_tier):
                        candidates.append((memory_id, memory_data))
                
            except Exception as e:
                print(f"Error evaluating migration candidate {memory_id}: {str(e)}")
                continue
        
        # Sort candidates by migration priority
        candidates.sort(key=lambda x: self._calculate_migration_priority(x[1]), reverse=True)
        
        return candidates
    
    async def _is_migration_eligible(self, memory_data: Dict, from_tier: str, to_tier: str) -> bool:
        """Check if memory is eligible for migration"""
        
        metadata = memory_data.get('metadata', {})
        content = memory_data.get('content', {})
        
        # Never migrate user preferences
        if metadata.get('category') == 'user_preferences':
            return False
        
        # Don't migrate recently accessed memories from hot tier
        if from_tier == 'hot':
            memory_id = metadata.get('id')
            if memory_id:
                last_access = self.memory.last_access.get(memory_id)
                if last_access and (datetime.now() - last_access).hours < 2:
                    return False
        
        # Don't migrate high-importance memories too quickly
        importance = metadata.get('importance', 0.5)
        if importance > 0.8 and from_tier == 'hot':
            return False
        
        # Don't migrate active patterns
        if content.get('type') == 'pattern' and to_tier == 'archive':
            # Check if pattern is still being used
            pattern_age_days = self._get_memory_age_hours(metadata) / 24
            if pattern_age_days < 30:  # Keep recent patterns active
                return False
        
        return True
    
    def _calculate_migration_priority(self, memory_data: Dict) -> float:
        """Calculate migration priority (higher = migrate first)"""
        
        priority = 0.0
        
        metadata = memory_data.get('metadata', {})
        content = memory_data.get('content', {})
        
        # Age factor (older = higher priority)
        age_hours = self._get_memory_age_hours(metadata)
        if age_hours:
            priority += min(age_hours / 168, 2.0)  # Cap at 2 weeks
        
        # Importance factor (lower importance = higher priority)
        importance = metadata.get('importance', 0.5)
        priority += (1.0 - importance)
        
        # Access frequency factor (less accessed = higher priority)
        memory_id = metadata.get('id')
        if memory_id:
            access_count = self.memory.access_counts.get(memory_id, 0)
            priority += max(0, 1.0 - (access_count / 10))
        
        # Success factor (failures migrate faster)
        if content.get('success') is False:
            priority += 0.5
        
        return priority
    
    async def _compress_memory(self, memory_data: Dict, compression_level: str) -> Dict:
        """Compress memory based on level"""
        
        if compression_level == 'none':
            return memory_data
        
        compressed_data = memory_data.copy()
        
        if compression_level == 'light':
            # Remove non-essential fields
            content = compressed_data.get('content', {})
            
            # Keep only essential trade data
            if content.get('type') == 'trade':
                essential_fields = [
                    'type', 'action', 'pool', 'success', 'profit',
                    'timestamp', 'gas_used'
                ]
                compressed_content = {
                    field: content[field] for field in essential_fields
                    if field in content
                }
                compressed_data['content'] = compressed_content
            
            # Compress metadata
            metadata = compressed_data.get('metadata', {})
            essential_metadata = [
                'id', 'timestamp', 'category', 'type', 'tier', 'importance'
            ]
            compressed_metadata = {
                field: metadata[field] for field in essential_metadata
                if field in metadata
            }
            compressed_metadata['compression_level'] = 'light'
            compressed_data['metadata'] = compressed_metadata
        
        elif compression_level == 'heavy':
            # Keep only summary information
            content = compressed_data.get('content', {})
            metadata = compressed_data.get('metadata', {})
            
            summary = self._create_content_summary(content)
            
            compressed_data = {
                'content': {
                    'type': content.get('type', 'unknown'),
                    'summary': summary,
                    'compressed': True
                },
                'metadata': {
                    'id': metadata.get('id'),
                    'timestamp': metadata.get('timestamp'),
                    'category': metadata.get('category'),
                    'tier': metadata.get('tier'),
                    'compression_level': 'heavy',
                    'original_size': len(json.dumps(memory_data))
                }
            }
        
        elif compression_level == 'pattern_only':
            # Extract only pattern information
            pattern = await self._extract_pattern_representation(memory_data)
            
            compressed_data = {
                'content': pattern,
                'metadata': {
                    'id': memory_data['metadata'].get('id'),
                    'original_timestamp': memory_data['metadata'].get('timestamp'),
                    'archived_at': datetime.now().isoformat(),
                    'compression_level': 'pattern_only',
                    'tier': 'archive'
                }
            }
        
        return compressed_data
    
    def _create_content_summary(self, content: Dict) -> str:
        """Create text summary of content"""
        
        content_type = content.get('type', 'unknown')
        
        if content_type == 'trade':
            action = content.get('action', 'unknown')
            pool = content.get('pool', 'unknown')
            success = content.get('success', False)
            profit = content.get('profit', 0)
            
            return f"{action} on {pool}: {'success' if success else 'failed'}, profit: {profit}"
        
        elif content_type == 'market_observation':
            pool = content.get('pool', 'unknown')
            observation = content.get('observation', 'N/A')
            
            return f"Market observation for {pool}: {observation}"
        
        elif content_type == 'pattern':
            pattern_type = content.get('pattern_type', 'unknown')
            success_rate = content.get('success_rate', 0)
            
            return f"Pattern {pattern_type} with {success_rate:.1%} success rate"
        
        else:
            return f"Content type: {content_type}"
    
    async def _extract_pattern_representation(self, memory_data: Dict) -> Dict:
        """Extract pattern representation from memory"""
        
        content = memory_data.get('content', {})
        
        if content.get('type') == 'trade':
            return {
                'type': 'trade_pattern',
                'action': content.get('action'),
                'pool': content.get('pool'),
                'success': content.get('success', False),
                'profit_category': self._categorize_profit(content.get('profit', 0)),
                'market_conditions': content.get('market_conditions', {}),
                'extracted_from': 'archive_compression'
            }
        
        elif content.get('type') == 'market_observation':
            return {
                'type': 'market_pattern',
                'pool': content.get('pool'),
                'observation_category': self._categorize_observation(content.get('observation', '')),
                'price_range': self._categorize_price(content.get('price')),
                'volume_category': self._categorize_volume(content.get('volume')),
                'extracted_from': 'archive_compression'
            }
        
        else:
            return {
                'type': 'generic_pattern',
                'original_type': content.get('type', 'unknown'),
                'summary': self._create_content_summary(content),
                'extracted_from': 'archive_compression'
            }
    
    def _categorize_profit(self, profit: float) -> str:
        """Categorize profit into buckets"""
        
        if profit > 100:
            return 'high_profit'
        elif profit > 10:
            return 'medium_profit'
        elif profit > 0:
            return 'low_profit'
        elif profit == 0:
            return 'break_even'
        elif profit > -10:
            return 'small_loss'
        else:
            return 'large_loss'
    
    def _categorize_observation(self, observation: str) -> str:
        """Categorize market observation"""
        
        observation_lower = observation.lower()
        
        if 'high' in observation_lower and 'volume' in observation_lower:
            return 'high_volume'
        elif 'low' in observation_lower and 'volume' in observation_lower:
            return 'low_volume'
        elif 'volatile' in observation_lower or 'volatility' in observation_lower:
            return 'high_volatility'
        elif 'stable' in observation_lower:
            return 'stable'
        elif 'trend' in observation_lower:
            return 'trending'
        else:
            return 'general'
    
    def _categorize_price(self, price: Optional[float]) -> str:
        """Categorize price into range"""
        
        if price is None:
            return 'unknown'
        
        # These thresholds would be configurable based on the token
        if price > 1000:
            return 'very_high'
        elif price > 100:
            return 'high'
        elif price > 10:
            return 'medium'
        elif price > 1:
            return 'low'
        else:
            return 'very_low'
    
    def _categorize_volume(self, volume: Optional[float]) -> str:
        """Categorize volume"""
        
        if volume is None:
            return 'unknown'
        
        # These thresholds would be configurable
        if volume > 1000000:
            return 'very_high'
        elif volume > 100000:
            return 'high'
        elif volume > 10000:
            return 'medium'
        elif volume > 1000:
            return 'low'
        else:
            return 'very_low'
    
    async def _extract_pattern_for_archive(self, memory_data: Dict) -> Optional[Dict]:
        """Extract pattern from memory before archiving"""
        
        content = memory_data.get('content', {})
        
        # Only extract patterns from suitable content types
        if content.get('type') not in ['trade', 'market_observation']:
            return None
        
        pattern = await self._extract_pattern_representation(memory_data)
        
        # Add archive-specific metadata
        pattern.update({
            'archived_from': memory_data['metadata'].get('id'),
            'original_timestamp': memory_data['metadata'].get('timestamp'),
            'archive_timestamp': datetime.now().isoformat(),
            'original_tier_path': f"{memory_data['metadata'].get('tier', 'unknown')}"
        })
        
        return pattern
    
    async def _archive_pattern(self, pattern: Dict):
        """Archive extracted pattern"""
        
        pattern_id = pattern.get('archived_from', f"pattern_{datetime.now().timestamp()}")
        
        self.memory.tiers['archive'][pattern_id] = {
            'content': pattern,
            'metadata': {
                'id': pattern_id,
                'type': 'archived_pattern',
                'category': 'patterns',
                'archived_at': datetime.now().isoformat(),
                'tier': 'archive',
                'compression_level': 'pattern_only'
            }
        }
    
    async def _update_storage_backend(
        self,
        memory_data: Dict,
        from_config: TierConfiguration,
        to_config: TierConfiguration
    ):
        """Update storage backend for migrated memory"""
        
        if from_config.storage_backend == to_config.storage_backend:
            return  # No backend change needed
        
        memory_id = memory_data['metadata']['id']
        
        try:
            # Save to new backend
            await self.storage_backends[to_config.storage_backend].store(
                memory_id, memory_data, to_config.name
            )
            
            # Remove from old backend if different
            if from_config.storage_backend != 'memory':
                await self.storage_backends[from_config.storage_backend].delete(
                    memory_id, from_config.name
                )
        
        except Exception as e:
            print(f"Error updating storage backend for {memory_id}: {str(e)}")
    
    def _get_memory_age_hours(self, metadata: Dict) -> Optional[float]:
        """Get memory age in hours"""
        
        timestamp_str = metadata.get('timestamp')
        if not timestamp_str:
            return None
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return (datetime.now() - timestamp).total_seconds() / 3600
        except ValueError:
            return None
    
    async def enforce_tier_limits(self) -> Dict[str, int]:
        """Enforce size limits on tiers"""
        
        enforcement_stats = {
            'evicted_memories': 0,
            'compressed_memories': 0,
            'errors': 0
        }
        
        for tier_name, tier_config in self.tier_configs.items():
            if tier_config.max_size == -1:
                continue  # Unlimited size
            
            tier_memories = self.memory.tiers[tier_name]
            current_size = len(tier_memories)
            
            if current_size > tier_config.max_size:
                # Need to evict or compress memories
                excess_count = current_size - tier_config.max_size
                
                # Get memories sorted by eviction priority
                eviction_candidates = await self._get_eviction_candidates(
                    tier_name, excess_count
                )
                
                for memory_id, memory_data in eviction_candidates:
                    try:
                        # Try to migrate first
                        next_tier = self._get_next_tier(tier_name)
                        if next_tier:
                            await self._migrate_memory_to_tier(
                                memory_id, memory_data, tier_name, next_tier
                            )
                        else:
                            # No next tier, remove from system
                            del tier_memories[memory_id]
                            enforcement_stats['evicted_memories'] += 1
                    
                    except Exception as e:
                        print(f"Error enforcing limit for {memory_id}: {str(e)}")
                        enforcement_stats['errors'] += 1
        
        return enforcement_stats
    
    async def _get_eviction_candidates(
        self,
        tier_name: str,
        count: int
    ) -> List[Tuple[str, Dict]]:
        """Get memories to evict from tier"""
        
        tier_memories = self.memory.tiers[tier_name]
        tier_config = self.tier_configs[tier_name]
        
        candidates = []
        
        for memory_id, memory_data in tier_memories.items():
            priority = self._calculate_eviction_priority(
                memory_data, tier_config.access_pattern
            )
            candidates.append((priority, memory_id, memory_data))
        
        # Sort by priority (higher = evict first)
        candidates.sort(reverse=True)
        
        # Return top candidates
        return [(memory_id, memory_data) for _, memory_id, memory_data in candidates[:count]]
    
    def _calculate_eviction_priority(self, memory_data: Dict, access_pattern: str) -> float:
        """Calculate eviction priority based on access pattern"""
        
        priority = 0.0
        metadata = memory_data.get('metadata', {})
        
        if access_pattern == 'lru':  # Least Recently Used
            memory_id = metadata.get('id')
            if memory_id:
                last_access = self.memory.last_access.get(memory_id)
                if last_access:
                    hours_since_access = (datetime.now() - last_access).total_seconds() / 3600
                    priority = hours_since_access  # Higher = evict first
                else:
                    priority = float('inf')  # Never accessed
        
        elif access_pattern == 'lfu':  # Least Frequently Used
            memory_id = metadata.get('id')
            if memory_id:
                access_count = self.memory.access_counts.get(memory_id, 0)
                priority = 1000 - access_count  # Lower access count = higher priority
        
        elif access_pattern == 'fifo':  # First In, First Out
            age_hours = self._get_memory_age_hours(metadata)
            priority = age_hours or 0
        
        # Adjust for importance
        importance = metadata.get('importance', 0.5)
        priority *= (1.0 - importance + 0.1)  # Higher importance = lower priority
        
        return priority
    
    def _get_next_tier(self, current_tier: str) -> Optional[str]:
        """Get the next tier in the hierarchy"""
        
        tier_order = ['hot', 'warm', 'cold', 'archive']
        
        try:
            current_index = tier_order.index(current_tier)
            if current_index < len(tier_order) - 1:
                return tier_order[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    async def _migrate_memory_to_tier(
        self,
        memory_id: str,
        memory_data: Dict,
        from_tier: str,
        to_tier: str
    ):
        """Migrate specific memory to target tier"""
        
        to_config = self.tier_configs[to_tier]
        
        # Apply compression if needed
        if to_config.compression_level != 'none':
            memory_data = await self._compress_memory(memory_data, to_config.compression_level)
        
        # Update metadata
        memory_data['metadata']['tier'] = to_tier
        memory_data['metadata']['migrated_at'] = datetime.now().isoformat()
        
        # Move to new tier
        self.memory.tiers[to_tier][memory_id] = memory_data
        
        # Remove from old tier
        del self.memory.tiers[from_tier][memory_id]
    
    async def get_tier_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tier statistics"""
        
        stats = {}
        
        for tier_name, tier_config in self.tier_configs.items():
            tier_memories = self.memory.tiers[tier_name]
            
            # Basic counts
            memory_count = len(tier_memories)
            
            # Size analysis
            total_size = 0
            compressed_count = 0
            
            for memory_data in tier_memories.values():
                # Estimate memory size
                memory_size = len(json.dumps(memory_data))
                total_size += memory_size
                
                # Count compressed memories
                compression_level = memory_data.get('metadata', {}).get('compression_level')
                if compression_level and compression_level != 'none':
                    compressed_count += 1
            
            # Age analysis
            ages = []
            for memory_data in tier_memories.values():
                age_hours = self._get_memory_age_hours(memory_data.get('metadata', {}))
                if age_hours is not None:
                    ages.append(age_hours)
            
            avg_age = sum(ages) / len(ages) if ages else 0
            min_age = min(ages) if ages else 0
            max_age = max(ages) if ages else 0
            
            # Category distribution
            categories = {}
            for memory_data in tier_memories.values():
                category = memory_data.get('metadata', {}).get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
            
            stats[tier_name] = {
                'config': asdict(tier_config),
                'current_size': memory_count,
                'utilization': memory_count / tier_config.max_size if tier_config.max_size > 0 else 0,
                'total_bytes': total_size,
                'compressed_memories': compressed_count,
                'compression_ratio': compressed_count / memory_count if memory_count > 0 else 0,
                'age_stats': {
                    'avg_age_hours': avg_age,
                    'min_age_hours': min_age,
                    'max_age_hours': max_age
                },
                'category_distribution': categories,
                'migration_candidates': len(await self._get_migration_candidates(
                    tier_name, 
                    self._get_next_tier(tier_name) or 'archive'
                )) if tier_config.auto_migration else 0
            }
        
        # Add global migration stats
        stats['migration_stats'] = self.migration_stats.copy()
        
        return stats
    
    async def optimize_tier_performance(self) -> Dict[str, Any]:
        """Optimize tier performance"""
        
        optimization_results = {
            'migrations_triggered': 0,
            'compressions_applied': 0,
            'evictions_performed': 0,
            'errors': 0
        }
        
        try:
            # Trigger migrations
            migration_results = await self.migrate_tiers()
            optimization_results['migrations_triggered'] = sum([
                migration_results.get('hot_to_warm', 0),
                migration_results.get('warm_to_cold', 0),
                migration_results.get('cold_to_archive', 0)
            ])
            
            # Enforce limits
            limit_results = await self.enforce_tier_limits()
            optimization_results['evictions_performed'] = limit_results.get('evicted_memories', 0)
            optimization_results['compressions_applied'] = limit_results.get('compressed_memories', 0)
            
            # Optimize access patterns
            await self._optimize_access_patterns()
            
        except Exception as e:
            print(f"Error optimizing tier performance: {str(e)}")
            optimization_results['errors'] += 1
        
        return optimization_results
    
    async def _optimize_access_patterns(self):
        """Optimize memory access patterns within tiers"""
        
        # This could implement more sophisticated caching strategies
        # For now, we'll just update access statistics
        
        current_time = datetime.now()
        
        for tier_name, tier_memories in self.memory.tiers.items():
            for memory_id, memory_data in tier_memories.items():
                # Update metadata with access pattern information
                metadata = memory_data.get('metadata', {})
                
                access_count = self.memory.access_counts.get(memory_id, 0)
                last_access = self.memory.last_access.get(memory_id)
                
                metadata['access_frequency'] = access_count
                if last_access:
                    hours_since_access = (current_time - last_access).total_seconds() / 3600
                    metadata['hours_since_last_access'] = hours_since_access


# Storage Backend Classes

class MemoryBackend:
    """In-memory storage backend"""
    
    def __init__(self):
        self.storage = {}
    
    async def store(self, memory_id: str, memory_data: Dict, tier: str):
        """Store memory in memory"""
        key = f"{tier}:{memory_id}"
        self.storage[key] = memory_data
    
    async def retrieve(self, memory_id: str, tier: str) -> Optional[Dict]:
        """Retrieve memory from memory"""
        key = f"{tier}:{memory_id}"
        return self.storage.get(key)
    
    async def delete(self, memory_id: str, tier: str):
        """Delete memory from memory"""
        key = f"{tier}:{memory_id}"
        self.storage.pop(key, None)


class DiskBackend:
    """Disk-based storage backend"""
    
    def __init__(self, base_path: str = './memory_storage'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def store(self, memory_id: str, memory_data: Dict, tier: str):
        """Store memory on disk"""
        tier_path = self.base_path / tier
        tier_path.mkdir(exist_ok=True)
        
        file_path = tier_path / f"{memory_id}.json.gz"
        
        # Compress and store
        json_data = json.dumps(memory_data).encode('utf-8')
        
        with gzip.open(file_path, 'wb') as f:
            f.write(json_data)
    
    async def retrieve(self, memory_id: str, tier: str) -> Optional[Dict]:
        """Retrieve memory from disk"""
        tier_path = self.base_path / tier
        file_path = tier_path / f"{memory_id}.json.gz"
        
        if not file_path.exists():
            return None
        
        try:
            with gzip.open(file_path, 'rb') as f:
                json_data = f.read().decode('utf-8')
                return json.loads(json_data)
        except Exception as e:
            print(f"Error retrieving {memory_id} from disk: {str(e)}")
            return None
    
    async def delete(self, memory_id: str, tier: str):
        """Delete memory from disk"""
        tier_path = self.base_path / tier
        file_path = tier_path / f"{memory_id}.json.gz"
        
        if file_path.exists():
            file_path.unlink()


class CloudBackend:
    """Cloud storage backend (placeholder)"""
    
    def __init__(self):
        self.enabled = False  # Disabled by default
    
    async def store(self, memory_id: str, memory_data: Dict, tier: str):
        """Store memory in cloud (placeholder)"""
        if not self.enabled:
            print(f"Cloud storage not configured, skipping storage of {memory_id}")
            return
        
        # Placeholder for cloud storage implementation
        pass
    
    async def retrieve(self, memory_id: str, tier: str) -> Optional[Dict]:
        """Retrieve memory from cloud (placeholder)"""
        if not self.enabled:
            return None
        
        # Placeholder for cloud retrieval
        return None
    
    async def delete(self, memory_id: str, tier: str):
        """Delete memory from cloud (placeholder)"""
        if not self.enabled:
            return
        
        # Placeholder for cloud deletion
        pass