"""
Intelligent memory pruning strategies.

This module implements age, relevance, and redundancy-based memory pruning
to maintain optimal memory system performance and prevent data bloat.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .system import MemorySystem


class MemoryPruning:
    """Intelligent memory pruning system"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.pruning_stats = {
            'total_pruned': 0,
            'by_age': 0,
            'by_relevance': 0,
            'by_redundancy': 0,
            'patterns_extracted': 0,
            'compression_applied': 0,
            'last_pruning': None
        }
        
        # Pruning configuration
        self.config = {
            'min_relevance_threshold': 0.3,
            'max_age_multiplier': 2.0,  # Allow 2x normal retention
            'similarity_threshold': 0.85,
            'min_access_threshold': 2,
            'cluster_min_samples': 2,
            'cluster_eps': 0.15,
            'batch_size': 100
        }
    
    async def execute_pruning_cycle(self) -> Dict[str, Any]:
        """Execute complete pruning cycle"""
        
        start_time = datetime.now()
        initial_count = self._get_total_memory_count()
        
        print(f"Starting pruning cycle with {initial_count} memories")
        
        # 1. Age-based pruning
        print("Executing age-based pruning...")
        age_pruned = await self.age_based_pruning()
        
        # 2. Relevance-based pruning
        print("Executing relevance-based pruning...")
        relevance_pruned = await self.relevance_based_pruning()
        
        # 3. Redundancy pruning
        print("Executing redundancy pruning...")
        redundancy_pruned = await self.redundancy_pruning()
        
        # 4. Pattern extraction before final pruning
        print("Extracting patterns...")
        patterns = await self.extract_patterns_before_pruning()
        
        # 5. Tier migration
        print("Migrating tiers...")
        migrations = await self.migrate_memories_during_pruning()
        
        # 6. Compression
        print("Applying compression...")
        compressed = await self.compress_similar_memories()
        
        # Calculate statistics
        final_count = self._get_total_memory_count()
        duration = (datetime.now() - start_time).total_seconds()
        
        # Update stats
        self.pruning_stats.update({
            'total_pruned': self.pruning_stats['total_pruned'] + (initial_count - final_count),
            'by_age': self.pruning_stats['by_age'] + age_pruned,
            'by_relevance': self.pruning_stats['by_relevance'] + relevance_pruned,
            'by_redundancy': self.pruning_stats['by_redundancy'] + redundancy_pruned,
            'patterns_extracted': self.pruning_stats['patterns_extracted'] + len(patterns),
            'compression_applied': self.pruning_stats['compression_applied'] + compressed,
            'last_pruning': datetime.now().isoformat()
        })
        
        result = {
            'duration': duration,
            'initial_count': initial_count,
            'final_count': final_count,
            'pruned': {
                'by_age': age_pruned,
                'by_relevance': relevance_pruned,
                'by_redundancy': redundancy_pruned
            },
            'patterns_extracted': len(patterns),
            'migrations': migrations,
            'compressed': compressed,
            'reduction_ratio': 1 - (final_count / initial_count) if initial_count > 0 else 0,
            'efficiency_score': self._calculate_efficiency_score(initial_count, final_count, duration)
        }
        
        print(f"Pruning completed: {initial_count} -> {final_count} memories ({result['reduction_ratio']:.2%} reduction)")
        
        return result
    
    async def age_based_pruning(self) -> int:
        """Prune memories based on age and tier"""
        
        pruned_count = 0
        current_time = datetime.now()
        
        # Process each tier with different age thresholds
        for tier_name, tier_memories in self.memory.tiers.items():
            if tier_name == 'archive':
                continue  # Don't prune archives
            
            # Create a list to avoid modifying dict during iteration
            memory_items = list(tier_memories.items())
            
            for memory_id, memory_data in memory_items:
                try:
                    metadata = memory_data.get('metadata', {})
                    
                    # Get memory age
                    age_hours = self._get_memory_age_hours(metadata)
                    if age_hours is None:
                        continue
                    
                    # Get category-specific retention period
                    category = metadata.get('category', 'market_observations')
                    retention_days = self.memory.categories.get_retention_period(category)
                    
                    # Skip permanent memories
                    if retention_days == -1:
                        continue
                    
                    # Calculate threshold with multiplier for safety
                    threshold_hours = retention_days * 24 * self.config['max_age_multiplier']
                    
                    should_prune = False
                    
                    if tier_name == 'hot' and age_hours > self.memory.config.hot_threshold:
                        # Move to warm or evaluate for pruning
                        if self._is_valuable_memory(memory_data):
                            await self._move_to_tier(memory_id, 'hot', 'warm')
                        elif age_hours > threshold_hours:
                            should_prune = True
                            
                    elif tier_name == 'warm' and age_hours > self.memory.config.warm_threshold:
                        # Move to cold or evaluate for pruning
                        if self._is_valuable_memory(memory_data):
                            await self._move_to_tier(memory_id, 'warm', 'cold')
                        elif age_hours > threshold_hours:
                            should_prune = True
                            
                    elif tier_name == 'cold' and age_hours > self.memory.config.cold_threshold:
                        # Extract pattern or prune
                        if age_hours > threshold_hours:
                            if self._can_extract_pattern(memory_data):
                                await self._extract_and_archive(memory_data)
                            should_prune = True
                    
                    if should_prune:
                        success = await self._prune_memory(memory_id, f"age_based_{tier_name}")
                        if success:
                            pruned_count += 1
                
                except Exception as e:
                    print(f"Error processing memory {memory_id} in age-based pruning: {str(e)}")
                    continue
        
        return pruned_count
    
    async def relevance_based_pruning(self) -> int:
        """Prune based on relevance and access patterns"""
        
        pruned_count = 0
        
        for tier_name, tier_memories in self.memory.tiers.items():
            if tier_name in ['archive', 'hot']:
                continue  # Don't prune archives or hot memories
            
            memory_items = list(tier_memories.items())
            
            for memory_id, memory_data in memory_items:
                try:
                    # Calculate relevance score
                    relevance_score = await self._calculate_relevance(memory_data)
                    
                    if relevance_score < self.config['min_relevance_threshold']:
                        # Check access pattern
                        access_count = self.memory.access_counts.get(memory_id, 0)
                        last_access = self.memory.last_access.get(memory_id)
                        
                        # If rarely accessed and low relevance, prune
                        if access_count < self.config['min_access_threshold']:
                            days_since_access = float('inf')
                            if last_access:
                                days_since_access = (datetime.now() - last_access).days
                            
                            # Prune if not accessed recently
                            if last_access is None or days_since_access > 7:
                                success = await self._prune_memory(memory_id, "low_relevance")
                                if success:
                                    pruned_count += 1
                
                except Exception as e:
                    print(f"Error processing memory {memory_id} in relevance-based pruning: {str(e)}")
                    continue
        
        return pruned_count
    
    async def redundancy_pruning(self) -> int:
        """Remove redundant and duplicate memories"""
        
        pruned_count = 0
        
        try:
            # Group memories by similarity
            memory_clusters = await self._cluster_similar_memories()
            
            for cluster in memory_clusters:
                if len(cluster) > 1:
                    # Sort by memory value (keep the best)
                    sorted_cluster = sorted(
                        cluster,
                        key=lambda m: self._calculate_memory_value(m),
                        reverse=True
                    )
                    
                    # Keep the best memory, prune the rest
                    best_memory = sorted_cluster[0]
                    
                    for memory_data in sorted_cluster[1:]:
                        memory_id = memory_data['metadata']['id']
                        
                        # Create merged memory from best + redundant info
                        await self._merge_redundant_memories(best_memory, memory_data)
                        
                        # Prune the redundant memory
                        success = await self._prune_memory(memory_id, "redundancy")
                        if success:
                            pruned_count += 1
        
        except Exception as e:
            print(f"Error in redundancy pruning: {str(e)}")
        
        return pruned_count
    
    async def extract_patterns_before_pruning(self) -> List[Dict]:
        """Extract patterns from memories before pruning them"""
        
        patterns = []
        
        try:
            # Extract patterns from each tier
            for tier_name, tier_memories in self.memory.tiers.items():
                if tier_name == 'archive':
                    continue
                
                tier_patterns = await self._extract_tier_patterns(tier_memories)
                patterns.extend(tier_patterns)
            
            # Store extracted patterns
            for pattern in patterns:
                await self.memory.add_memory(
                    content=pattern,
                    metadata={'type': 'pattern', 'category': 'patterns', 'source': 'pruning_extraction'}
                )
        
        except Exception as e:
            print(f"Error extracting patterns: {str(e)}")
        
        return patterns
    
    async def migrate_memories_during_pruning(self) -> Dict[str, int]:
        """Migrate memories between tiers during pruning"""
        
        migration_counts = {
            'hot_to_warm': 0,
            'warm_to_cold': 0,
            'cold_to_archive': 0
        }
        
        try:
            current_time = datetime.now()
            
            # Hot to Warm migration
            for memory_id, memory_data in list(self.memory.tiers['hot'].items()):
                age_hours = self._get_memory_age_hours(memory_data.get('metadata', {}))
                
                if age_hours and age_hours > self.memory.config.hot_threshold:
                    await self._move_to_tier(memory_id, 'hot', 'warm')
                    migration_counts['hot_to_warm'] += 1
            
            # Warm to Cold migration
            for memory_id, memory_data in list(self.memory.tiers['warm'].items()):
                age_hours = self._get_memory_age_hours(memory_data.get('metadata', {}))
                
                if age_hours and age_hours > self.memory.config.warm_threshold:
                    await self._move_to_tier(memory_id, 'warm', 'cold')
                    migration_counts['warm_to_cold'] += 1
            
            # Cold to Archive migration (pattern extraction)
            for memory_id, memory_data in list(self.memory.tiers['cold'].items()):
                age_hours = self._get_memory_age_hours(memory_data.get('metadata', {}))
                
                if age_hours and age_hours > self.memory.config.cold_threshold:
                    # Extract pattern and archive
                    pattern = await self._extract_pattern(memory_data)
                    if pattern:
                        await self._archive_pattern(pattern)
                        migration_counts['cold_to_archive'] += 1
                    
                    # Remove from cold storage
                    del self.memory.tiers['cold'][memory_id]
        
        except Exception as e:
            print(f"Error in tier migration: {str(e)}")
        
        return migration_counts
    
    async def compress_similar_memories(self) -> int:
        """Compress similar memories into more compact representations"""
        
        compressed_count = 0
        
        try:
            # Find compressible memory groups
            compressible_groups = await self._find_compressible_memories()
            
            for group in compressible_groups:
                if len(group) >= 3:  # Only compress groups of 3+
                    # Create compressed representation
                    compressed_memory = await self._create_compressed_memory(group)
                    
                    # Add compressed memory
                    await self.memory.add_memory(
                        content=compressed_memory,
                        metadata={
                            'type': 'compressed',
                            'category': group[0]['metadata'].get('category', 'market_observations'),
                            'compression_level': 'group',
                            'original_count': len(group)
                        }
                    )
                    
                    # Remove original memories
                    for memory_data in group:
                        memory_id = memory_data['metadata']['id']
                        await self._prune_memory(memory_id, "compression")
                        compressed_count += 1
        
        except Exception as e:
            print(f"Error in memory compression: {str(e)}")
        
        return compressed_count
    
    def _get_total_memory_count(self) -> int:
        """Get total count of memories across all tiers"""
        return sum(len(tier) for tier in self.memory.tiers.values())
    
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
    
    def _is_valuable_memory(self, memory_data: Dict) -> bool:
        """Determine if memory is valuable enough to keep longer"""
        
        metadata = memory_data.get('metadata', {})
        content = memory_data.get('content', {})
        
        # High importance memories are valuable
        importance = metadata.get('importance', 0.5)
        if importance > 0.8:
            return True
        
        # Frequently accessed memories are valuable
        memory_id = metadata.get('id')
        if memory_id:
            access_count = self.memory.access_counts.get(memory_id, 0)
            if access_count > 5:
                return True
        
        # Successful trades are valuable
        if content.get('type') == 'trade' and content.get('success', False):
            if content.get('profit', 0) > 0:
                return True
        
        # Patterns are always valuable
        if content.get('type') == 'pattern':
            return True
        
        # User preferences are valuable
        if metadata.get('category') == 'user_preferences':
            return True
        
        return False
    
    def _can_extract_pattern(self, memory_data: Dict) -> bool:
        """Check if memory can contribute to pattern extraction"""
        
        content = memory_data.get('content', {})
        metadata = memory_data.get('metadata', {})
        
        # Skip if already a pattern
        if content.get('type') == 'pattern':
            return False
        
        # Check if category is pattern-eligible
        category = metadata.get('category', 'market_observations')
        if not self.memory.categories.is_pattern_eligible(category):
            return False
        
        # Must have sufficient structure
        required_fields = ['type', 'action'] if content.get('type') == 'trade' else ['type']
        
        return all(field in content for field in required_fields)
    
    async def _calculate_relevance(self, memory_data: Dict) -> float:
        """Calculate relevance score for memory"""
        
        score = 0.5  # Base score
        
        metadata = memory_data.get('metadata', {})
        content = memory_data.get('content', {})
        
        # Category importance
        category = metadata.get('category', 'market_observations')
        category_importance = self.memory.categories.get_importance(category)
        score *= category_importance
        
        # Success factor (for trades)
        if content.get('success', False):
            score *= 1.2
        
        # Profit factor
        profit = content.get('profit', 0)
        if profit > 0:
            score *= (1 + min(profit / 100, 0.5))  # Cap at 50% boost
        
        # Access frequency
        memory_id = metadata.get('id')
        if memory_id:
            access_count = self.memory.access_counts.get(memory_id, 0)
            score *= (1 + min(access_count / 10, 0.3))  # Cap at 30% boost
        
        # Recency factor
        age_hours = self._get_memory_age_hours(metadata)
        if age_hours is not None:
            age_days = age_hours / 24
            recency_factor = max(0.5, 1 - (age_days / 30))  # Decay over 30 days
            score *= recency_factor
        
        # Pattern contribution potential
        if self._can_extract_pattern(memory_data):
            score *= 1.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _cluster_similar_memories(self) -> List[List[Dict]]:
        """Cluster similar memories using embeddings"""
        
        all_memories = []
        embeddings = []
        
        try:
            # Collect memories and embeddings from non-archive tiers
            for tier_name, tier_memories in self.memory.tiers.items():
                if tier_name == 'archive':
                    continue
                
                for memory_id, memory_data in tier_memories.items():
                    all_memories.append(memory_data)
                    
                    # Get or generate embedding
                    embedding = await self._get_or_generate_embedding(memory_data)
                    if embedding is not None:
                        embeddings.append(embedding)
                    else:
                        all_memories.pop()  # Remove memory if no embedding
            
            if len(embeddings) < 2:
                return []
            
            # Use DBSCAN for clustering
            embeddings_array = np.array(embeddings)
            
            clustering = DBSCAN(
                eps=self.config['cluster_eps'],
                min_samples=self.config['cluster_min_samples'],
                metric='cosine'
            )
            
            labels = clustering.fit_predict(embeddings_array)
            
            # Group memories by cluster
            clusters = {}
            for idx, label in enumerate(labels):
                if label != -1:  # Ignore noise points
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(all_memories[idx])
            
            return list(clusters.values())
        
        except Exception as e:
            print(f"Error clustering memories: {str(e)}")
            return []
    
    async def _get_or_generate_embedding(self, memory_data: Dict) -> Optional[np.ndarray]:
        """Get or generate embedding for memory"""
        
        try:
            # Check if embedding already exists
            embedding = memory_data.get('embedding')
            if embedding is not None:
                return np.array(embedding)
            
            # Generate embedding using Mem0's embedder
            content_text = self.memory.format_memory(memory_data['content'])
            
            # This would use Mem0's embedding system
            # For now, we'll create a simple hash-based embedding
            embedding = self._create_simple_embedding(content_text)
            
            return embedding
        
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def _create_simple_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding from text (fallback method)"""
        
        # Simple hash-based embedding for demonstration
        # In production, this would use proper embeddings
        
        words = text.lower().split()
        embedding = np.zeros(128)  # Simple 128-dim embedding
        
        for i, word in enumerate(words[:32]):  # Use first 32 words
            word_hash = hash(word) % 128
            embedding[word_hash] += 1.0
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _calculate_memory_value(self, memory_data: Dict) -> float:
        """Calculate overall value of memory for ranking"""
        
        metadata = memory_data.get('metadata', {})
        content = memory_data.get('content', {})
        
        value = 0.0
        
        # Base importance
        value += metadata.get('importance', 0.5) * 0.3
        
        # Access frequency
        memory_id = metadata.get('id')
        if memory_id:
            access_count = self.memory.access_counts.get(memory_id, 0)
            value += min(access_count / 10, 0.2)  # Up to 20% from access
        
        # Success bonus
        if content.get('success', False):
            value += 0.2
        
        # Profit bonus
        profit = content.get('profit', 0)
        if profit > 0:
            value += min(profit / 1000, 0.3)  # Up to 30% from profit
        
        # Recency bonus
        age_hours = self._get_memory_age_hours(metadata)
        if age_hours is not None:
            age_days = age_hours / 24
            recency_factor = max(0.0, 1 - (age_days / 30))  # Linear decay over 30 days
            value += recency_factor * 0.1  # Up to 10% from recency
        
        return value
    
    async def _move_to_tier(self, memory_id: str, from_tier: str, to_tier: str):
        """Move memory between tiers"""
        
        try:
            memory_data = self.memory.tiers[from_tier].get(memory_id)
            if not memory_data:
                return
            
            # Update metadata
            memory_data['metadata']['tier'] = to_tier
            memory_data['metadata']['migrated_at'] = datetime.now().isoformat()
            
            # Move to new tier
            self.memory.tiers[to_tier][memory_id] = memory_data
            
            # Remove from old tier
            del self.memory.tiers[from_tier][memory_id]
        
        except Exception as e:
            print(f"Error moving memory {memory_id} from {from_tier} to {to_tier}: {str(e)}")
    
    async def _prune_memory(self, memory_id: str, reason: str = "unknown") -> bool:
        """Prune memory from system"""
        
        try:
            # Remove from all tiers
            for tier_name, tier_data in self.memory.tiers.items():
                if memory_id in tier_data:
                    del tier_data[memory_id]
                    break
            
            # Clean up access tracking
            self.memory.access_counts.pop(memory_id, None)
            self.memory.last_access.pop(memory_id, None)
            
            # Try to remove from Mem0
            try:
                self.memory.mem0.delete(memory_id=memory_id)
            except Exception as e:
                print(f"Warning: Could not delete from Mem0: {str(e)}")
            
            return True
        
        except Exception as e:
            print(f"Error pruning memory {memory_id}: {str(e)}")
            return False
    
    async def _extract_and_archive(self, memory_data: Dict):
        """Extract pattern from memory and archive it"""
        
        try:
            pattern = await self._extract_pattern(memory_data)
            if pattern:
                await self._archive_pattern(pattern)
        except Exception as e:
            print(f"Error extracting and archiving pattern: {str(e)}")
    
    async def _extract_pattern(self, memory_data: Dict) -> Optional[Dict]:
        """Extract pattern from memory data"""
        
        content = memory_data.get('content', {})
        
        if content.get('type') == 'trade':
            return {
                'type': 'pattern',
                'pattern_type': 'trade_pattern',
                'action': content.get('action'),
                'pool': content.get('pool'),
                'success': content.get('success', False),
                'profit': content.get('profit', 0),
                'market_conditions': content.get('market_conditions', {}),
                'extracted_from': memory_data['metadata']['id'],
                'extracted_at': datetime.now().isoformat()
            }
        
        return None
    
    async def _archive_pattern(self, pattern: Dict):
        """Archive extracted pattern"""
        
        self.memory.tiers['archive'][pattern['extracted_from']] = {
            'content': pattern,
            'metadata': {
                'id': pattern['extracted_from'],
                'type': 'pattern',
                'category': 'patterns',
                'archived_at': datetime.now().isoformat(),
                'tier': 'archive'
            }
        }
    
    async def _merge_redundant_memories(self, best_memory: Dict, redundant_memory: Dict):
        """Merge information from redundant memories into the best one"""
        
        try:
            best_content = best_memory.get('content', {})
            redundant_content = redundant_memory.get('content', {})
            
            # Merge occurrence count
            best_content['occurrence_count'] = best_content.get('occurrence_count', 1) + 1
            
            # Update confidence if applicable
            if 'confidence' in redundant_content:
                best_confidence = best_content.get('confidence', 0.5)
                redundant_confidence = redundant_content.get('confidence', 0.5)
                best_content['confidence'] = max(best_confidence, redundant_confidence)
            
            # Add merged flag
            best_memory['metadata']['merged_count'] = best_memory['metadata'].get('merged_count', 0) + 1
            best_memory['metadata']['last_merge'] = datetime.now().isoformat()
        
        except Exception as e:
            print(f"Error merging memories: {str(e)}")
    
    async def _extract_tier_patterns(self, tier_memories: Dict) -> List[Dict]:
        """Extract patterns from memories in a tier"""
        
        patterns = []
        
        # Group memories by type for pattern extraction
        type_groups = {}
        
        for memory_data in tier_memories.values():
            content = memory_data.get('content', {})
            memory_type = content.get('type', 'general')
            
            if memory_type not in type_groups:
                type_groups[memory_type] = []
            
            type_groups[memory_type].append(memory_data)
        
        # Extract patterns from each type group
        for memory_type, memories in type_groups.items():
            if len(memories) >= 3:  # Need at least 3 for pattern
                type_patterns = await self._extract_type_patterns(memory_type, memories)
                patterns.extend(type_patterns)
        
        return patterns
    
    async def _extract_type_patterns(self, memory_type: str, memories: List[Dict]) -> List[Dict]:
        """Extract patterns from memories of same type"""
        
        patterns = []
        
        if memory_type == 'trade':
            # Group by action and pool
            action_groups = {}
            
            for memory_data in memories:
                content = memory_data.get('content', {})
                action = content.get('action', 'unknown')
                pool = content.get('pool', 'unknown')
                key = f"{action}_{pool}"
                
                if key not in action_groups:
                    action_groups[key] = []
                
                action_groups[key].append(memory_data)
            
            # Extract patterns from groups with enough data
            for key, group in action_groups.items():
                if len(group) >= 3:
                    pattern = await self._create_trade_pattern(group)
                    if pattern:
                        patterns.append(pattern)
        
        return patterns
    
    async def _create_trade_pattern(self, trade_memories: List[Dict]) -> Optional[Dict]:
        """Create trade pattern from group of trade memories"""
        
        if not trade_memories:
            return None
        
        # Calculate statistics
        successes = [m for m in trade_memories if m['content'].get('success', False)]
        success_rate = len(successes) / len(trade_memories)
        
        profits = [m['content'].get('profit', 0) for m in trade_memories]
        avg_profit = np.mean(profits) if profits else 0
        
        # Extract common characteristics
        first_trade = trade_memories[0]['content']
        
        pattern = {
            'type': 'pattern',
            'pattern_type': 'trade_pattern',
            'action': first_trade.get('action'),
            'pool': first_trade.get('pool'),
            'occurrence_count': len(trade_memories),
            'success_rate': success_rate,
            'avg_profit': avg_profit,
            'confidence': min(success_rate * (len(trade_memories) / 10), 1.0),
            'discovered_at': datetime.now().isoformat()
        }
        
        return pattern
    
    async def _find_compressible_memories(self) -> List[List[Dict]]:
        """Find groups of memories that can be compressed together"""
        
        compressible_groups = []
        
        try:
            # Look for similar memories in warm and cold tiers
            for tier_name in ['warm', 'cold']:
                tier_memories = list(self.memory.tiers[tier_name].values())
                
                if len(tier_memories) < 3:
                    continue
                
                # Group by category and type
                groups = {}
                
                for memory_data in tier_memories:
                    metadata = memory_data.get('metadata', {})
                    content = memory_data.get('content', {})
                    
                    category = metadata.get('category', 'general')
                    memory_type = content.get('type', 'general')
                    key = f"{category}_{memory_type}"
                    
                    if key not in groups:
                        groups[key] = []
                    
                    groups[key].append(memory_data)
                
                # Find groups large enough for compression
                for group in groups.values():
                    if len(group) >= 5:  # Need at least 5 for compression
                        compressible_groups.append(group)
        
        except Exception as e:
            print(f"Error finding compressible memories: {str(e)}")
        
        return compressible_groups
    
    async def _create_compressed_memory(self, memory_group: List[Dict]) -> Dict:
        """Create compressed representation of memory group"""
        
        if not memory_group:
            return {}
        
        # Extract common characteristics
        first_memory = memory_group[0]
        category = first_memory['metadata'].get('category', 'general')
        memory_type = first_memory['content'].get('type', 'general')
        
        # Calculate aggregates
        total_count = len(memory_group)
        success_count = sum(1 for m in memory_group if m['content'].get('success', False))
        success_rate = success_count / total_count
        
        profits = [m['content'].get('profit', 0) for m in memory_group]
        avg_profit = np.mean(profits) if profits else 0
        total_profit = sum(profits)
        
        # Time range
        timestamps = []
        for memory in memory_group:
            timestamp_str = memory['metadata'].get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except ValueError:
                    continue
        
        time_range = None
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            time_range = {
                'start': min_time.isoformat(),
                'end': max_time.isoformat(),
                'duration_days': (max_time - min_time).days
            }
        
        # Create compressed memory
        compressed = {
            'type': 'compressed_group',
            'original_type': memory_type,
            'category': category,
            'total_memories': total_count,
            'success_rate': success_rate,
            'avg_profit': avg_profit,
            'total_profit': total_profit,
            'time_range': time_range,
            'compression_ratio': 1.0 / total_count,  # Compression achieved
            'compressed_at': datetime.now().isoformat(),
            'sample_memory': first_memory['content']  # Keep one example
        }
        
        return compressed
    
    def _calculate_efficiency_score(self, initial_count: int, final_count: int, duration: float) -> float:
        """Calculate pruning efficiency score"""
        
        if initial_count == 0:
            return 1.0
        
        # Base efficiency from reduction ratio
        reduction_ratio = 1 - (final_count / initial_count)
        
        # Time efficiency (prefer faster pruning)
        time_efficiency = max(0.1, 1.0 / (1 + duration / 60))  # Normalize by minutes
        
        # Combined score
        efficiency = (reduction_ratio * 0.7) + (time_efficiency * 0.3)
        
        return min(efficiency, 1.0)
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get pruning statistics"""
        return self.pruning_stats.copy()
    
    def reset_pruning_stats(self):
        """Reset pruning statistics"""
        self.pruning_stats = {
            'total_pruned': 0,
            'by_age': 0,
            'by_relevance': 0,
            'by_redundancy': 0,
            'patterns_extracted': 0,
            'compression_applied': 0,
            'last_pruning': None
        }