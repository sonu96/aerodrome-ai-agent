"""
Core memory operations with Mem0 integration.

This module provides CRUD operations for memories, including storage,
retrieval, search, and updates through Mem0's semantic memory system.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from .system import MemorySystem


class MemoryOperations:
    """Core Mem0 operations wrapper"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.user_id = "me"  # Single user optimization
    
    async def add_memory(
        self,
        content: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add memory to Mem0 with metadata"""
        
        # Generate unique ID
        memory_id = self.memory.generate_memory_id(content)
        
        # Prepare memory content
        memory_text = self.memory.format_memory(content)
        
        # Add metadata
        if metadata is None:
            metadata = {}
        
        # Enrich metadata with system information
        enriched_metadata = {
            'id': memory_id,
            'timestamp': datetime.now().isoformat(),
            'type': content.get('type', 'general'),
            'category': self.memory.categorize_memory(content),
            'ttl': self.memory.calculate_ttl(content),
            'access_count': 0,
            'tier': 'hot',
            'importance': self._calculate_importance(content),
            **metadata  # User-provided metadata takes precedence
        }
        
        # Store in Mem0
        try:
            result = self.memory.mem0.add(
                memory_text,
                user_id=self.user_id,
                metadata=enriched_metadata
            )
            
            # Extract memory ID from Mem0 result if available
            if isinstance(result, dict) and 'id' in result:
                memory_id = result['id']
            elif hasattr(result, 'id'):
                memory_id = result.id
            
        except Exception as e:
            raise RuntimeError(f"Failed to add memory to Mem0: {str(e)}")
        
        # Track in local tier
        self.memory.tiers['hot'][memory_id] = {
            'content': content,
            'metadata': enriched_metadata,
            'embedding': None,  # Will be set by Mem0
            'created_at': datetime.now().isoformat()
        }
        
        # Initialize access tracking
        self.memory.access_counts[memory_id] = 0
        self.memory.last_access[memory_id] = datetime.now()
        
        return memory_id
    
    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search memories using Mem0's semantic search"""
        
        try:
            # Search in Mem0
            results = self.memory.mem0.search(
                query=query,
                user_id=self.user_id,
                limit=limit * 2  # Get more for filtering
            )
            
            # Convert to standard format if needed
            if isinstance(results, dict) and 'results' in results:
                results = results['results']
            elif not isinstance(results, list):
                results = []
            
            # Apply additional filters
            if filters:
                results = self._apply_filters(results, filters)
            
            # Update access tracking
            for result in results:
                memory_id = self._extract_memory_id(result)
                if memory_id:
                    self.memory.access_counts[memory_id] = self.memory.access_counts.get(memory_id, 0) + 1
                    self.memory.last_access[memory_id] = datetime.now()
            
            # Sort by relevance and recency
            results = self._rank_results(results, query)
            
            return results[:limit]
            
        except Exception as e:
            raise RuntimeError(f"Failed to search memories: {str(e)}")
    
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update existing memory"""
        
        try:
            # Get current memory
            current = await self.get_memory_by_id(memory_id)
            if not current:
                return False
            
            # Merge updates
            updated_content = {**current['content'], **updates}
            updated_metadata = current['metadata'].copy()
            updated_metadata.update({
                'last_modified': datetime.now().isoformat(),
                'modification_count': updated_metadata.get('modification_count', 0) + 1
            })
            
            # Update in Mem0
            updated_text = self.memory.format_memory(updated_content)
            
            result = self.memory.mem0.update(
                memory_id=memory_id,
                data=updated_text,
                metadata=updated_metadata
            )
            
            # Update in tier
            tier = updated_metadata.get('tier', 'hot')
            if memory_id in self.memory.tiers[tier]:
                self.memory.tiers[tier][memory_id].update({
                    'content': updated_content,
                    'metadata': updated_metadata,
                    'updated_at': datetime.now().isoformat()
                })
            
            return True
            
        except Exception as e:
            print(f"Failed to update memory {memory_id}: {str(e)}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from the system"""
        
        try:
            # Delete from Mem0
            self.memory.mem0.delete(memory_id=memory_id)
            
            # Remove from all tiers
            for tier_name, tier_data in self.memory.tiers.items():
                if memory_id in tier_data:
                    del tier_data[memory_id]
                    break
            
            # Clean up access tracking
            self.memory.access_counts.pop(memory_id, None)
            self.memory.last_access.pop(memory_id, None)
            
            return True
            
        except Exception as e:
            print(f"Failed to delete memory {memory_id}: {str(e)}")
            return False
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory by ID"""
        
        # Check tiers first (faster)
        for tier_data in self.memory.tiers.values():
            if memory_id in tier_data:
                return tier_data[memory_id]
        
        # If not in tiers, search Mem0
        try:
            results = await self.search_memories(f"id:{memory_id}", limit=1)
            return results[0] if results else None
        except Exception as e:
            print(f"Failed to get memory {memory_id}: {str(e)}")
            return None
    
    async def store_trade_result(
        self,
        action: Dict,
        result: Dict,
        market_context: Dict
    ) -> str:
        """Store trade execution result"""
        
        trade_memory = {
            'type': 'trade',
            'action': action.get('type', action.get('action')),
            'pool': action.get('pool'),
            'tokens': action.get('tokens', []),
            'amounts': action.get('amounts', []),
            'timestamp': datetime.now().isoformat(),
            'success': result.get('success', False),
            'tx_hash': result.get('tx_hash'),
            'gas_used': result.get('gas_used', 0),
            'profit': self._calculate_profit(action, result),
            'market_context': market_context,
            'decision_confidence': action.get('confidence', 0.5),
            'error_message': result.get('error') if not result.get('success', False) else None
        }
        
        # Store in memory
        memory_id = await self.add_memory(
            content=trade_memory,
            metadata={
                'category': 'trades',
                'importance': 0.9 if trade_memory['success'] else 0.7,
                'searchable_tags': self._create_trade_tags(trade_memory)
            }
        )
        
        # Check for pattern emergence in background
        asyncio.create_task(self._check_pattern_emergence(trade_memory))
        
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
            if (m.get('content', {}).get('pool') == pool and
                m.get('content', {}).get('action') == action_type)
        ]
        
        # Sort by recency and success
        similar.sort(
            key=lambda m: (
                m.get('content', {}).get('success', False),
                m.get('metadata', {}).get('timestamp', '')
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
        query_parts = ["trade"]
        if pool:
            query_parts.append(f"pool {pool}")
        if action_type:
            query_parts.append(action_type)
        
        query = " ".join(query_parts)
        
        # Get relevant memories
        memories = await self.search_memories(
            query=query,
            limit=1000,
            filters={'category': 'trades'}
        )
        
        # Filter by time window
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_memories = []
        
        for memory in memories:
            timestamp_str = memory.get('metadata', {}).get('timestamp')
            if timestamp_str:
                try:
                    memory_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if memory_date > cutoff_date:
                        recent_memories.append(memory)
                except ValueError:
                    continue  # Skip memories with invalid timestamps
        
        if not recent_memories:
            return 0.5  # Default to 50% if no data
        
        # Calculate success rate
        successes = sum(
            1 for m in recent_memories
            if m.get('content', {}).get('success', False)
        )
        
        return successes / len(recent_memories)
    
    def _calculate_importance(self, content: Dict) -> float:
        """Calculate importance score for memory"""
        
        base_importance = 0.5
        
        # Trade memories are more important
        if content.get('type') == 'trade':
            base_importance = 0.8
            
            # Successful trades are even more important
            if content.get('success', False):
                base_importance = 0.9
            
            # High profit trades are very important
            profit = content.get('profit', 0)
            if profit > 0:
                base_importance = min(1.0, base_importance + (profit / 1000))  # Cap at 1.0
        
        # Pattern memories are always important
        elif content.get('type') == 'pattern':
            base_importance = 1.0
        
        # Error memories are important for learning
        elif content.get('type') == 'error':
            base_importance = 0.8
        
        return base_importance
    
    def _apply_filters(self, results: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply additional filters to search results"""
        
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            content = result.get('content', {})
            
            # Apply each filter
            include_result = True
            
            for filter_key, filter_value in filters.items():
                if filter_key == 'category':
                    if metadata.get('category') != filter_value:
                        include_result = False
                        break
                
                elif filter_key == 'type':
                    if metadata.get('type') != filter_value:
                        include_result = False
                        break
                
                elif filter_key == 'tier':
                    if metadata.get('tier') != filter_value:
                        include_result = False
                        break
                
                elif filter_key == 'success':
                    if content.get('success') != filter_value:
                        include_result = False
                        break
                
                elif filter_key == 'min_importance':
                    if metadata.get('importance', 0) < filter_value:
                        include_result = False
                        break
                
                elif filter_key == 'max_age_days':
                    timestamp_str = metadata.get('timestamp')
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            age_days = (datetime.now() - timestamp).days
                            if age_days > filter_value:
                                include_result = False
                                break
                        except ValueError:
                            pass
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def _rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Sort results by relevance and recency"""
        
        def calculate_score(result: Dict) -> float:
            metadata = result.get('metadata', {})
            content = result.get('content', {})
            
            # Base relevance score (from Mem0)
            score = result.get('score', 0.5)
            
            # Recency boost
            timestamp_str = metadata.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    age_days = (datetime.now() - timestamp).days
                    recency_factor = max(0.1, 1.0 - (age_days / 30))  # Decay over 30 days
                    score *= recency_factor
                except ValueError:
                    pass
            
            # Importance boost
            importance = metadata.get('importance', 0.5)
            score *= (0.5 + importance)  # Scale by importance
            
            # Access frequency boost
            memory_id = metadata.get('id')
            if memory_id:
                access_count = self.memory.access_counts.get(memory_id, 0)
                access_boost = min(0.3, access_count / 10)  # Cap at 30% boost
                score *= (1.0 + access_boost)
            
            # Success boost for trades
            if content.get('success', False):
                score *= 1.2
            
            return score
        
        # Sort by calculated score
        results.sort(key=calculate_score, reverse=True)
        return results
    
    def _extract_memory_id(self, result: Dict) -> Optional[str]:
        """Extract memory ID from search result"""
        
        # Try different possible locations for memory ID
        memory_id = None
        
        if 'id' in result:
            memory_id = result['id']
        elif 'metadata' in result and 'id' in result['metadata']:
            memory_id = result['metadata']['id']
        elif hasattr(result, 'id'):
            memory_id = result.id
        
        return memory_id
    
    def _calculate_profit(self, action: Dict, result: Dict) -> float:
        """Calculate profit from trade action and result"""
        
        # Extract profit information from result
        profit = result.get('profit', 0)
        
        # If not provided, try to calculate from amounts
        if profit == 0:
            input_amount = action.get('amount_in', 0)
            output_amount = result.get('amount_out', 0)
            
            if input_amount > 0 and output_amount > 0:
                profit = output_amount - input_amount
        
        return float(profit)
    
    def _create_trade_tags(self, trade_memory: Dict) -> List[str]:
        """Create searchable tags for trade memory"""
        
        tags = []
        
        # Action tag
        action = trade_memory.get('action')
        if action:
            tags.append(action)
        
        # Pool tag
        pool = trade_memory.get('pool')
        if pool:
            tags.append(f"pool_{pool}")
        
        # Success/failure tag
        success = trade_memory.get('success', False)
        tags.append('successful' if success else 'failed')
        
        # Token tags
        tokens = trade_memory.get('tokens', [])
        for token in tokens:
            if token:
                tags.append(f"token_{token}")
        
        # Profit category
        profit = trade_memory.get('profit', 0)
        if profit > 0:
            tags.append('profitable')
        elif profit < 0:
            tags.append('loss')
        else:
            tags.append('break_even')
        
        return tags
    
    async def _check_pattern_emergence(self, trade_memory: Dict):
        """Check if this trade contributes to an emerging pattern"""
        
        try:
            # Look for similar trades in the past
            similar_trades = await self.recall_similar_trades(
                pool=trade_memory.get('pool'),
                action_type=trade_memory.get('action'),
                limit=10
            )
            
            # If we have enough similar trades, consider pattern extraction
            if len(similar_trades) >= 3:
                # This would trigger pattern extraction
                # For now, just log it
                print(f"Pattern potential detected for {trade_memory.get('action')} on {trade_memory.get('pool')}")
        
        except Exception as e:
            print(f"Error checking pattern emergence: {str(e)}")