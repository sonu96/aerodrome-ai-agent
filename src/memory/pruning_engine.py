"""
Sophisticated memory pruning engine for the Aerodrome AI Agent.

This module provides a comprehensive memory pruning system that works with Mem0's
latest features including graph memory, tiered pruning strategies, memory consolidation,
and intelligent cleanup based on confidence scores.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import math

from .mem0_client import EnhancedMem0Client
from .memory_categories import MemoryCategory, MemoryCategoryConfig, MemoryMetadata

logger = logging.getLogger(__name__)


@dataclass
class PruningStats:
    """Statistics from pruning operations."""
    total_memories_before: int
    total_memories_after: int
    memories_deleted: int
    memories_consolidated: int
    categories_processed: int
    processing_time_seconds: float
    errors: List[str]
    
    @property
    def deletion_rate(self) -> float:
        """Calculate deletion rate percentage."""
        if self.total_memories_before == 0:
            return 0.0
        return (self.memories_deleted / self.total_memories_before) * 100
    
    @property
    def consolidation_rate(self) -> float:
        """Calculate consolidation rate percentage."""
        if self.total_memories_before == 0:
            return 0.0
        return (self.memories_consolidated / self.total_memories_before) * 100


@dataclass
class ConsolidationCandidate:
    """Represents a group of similar memories that can be consolidated."""
    memories: List[Dict[str, Any]]
    similarity_score: float
    consolidated_content: str
    combined_confidence: float
    combined_metadata: Dict[str, Any]


class MemoryPruningEngine:
    """Sophisticated memory pruning engine with tiered strategies."""
    
    def __init__(
        self,
        mem0_client: EnhancedMem0Client,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pruning engine.
        
        Args:
            mem0_client: Enhanced Mem0 client instance
            config: Engine configuration
        """
        self.client = mem0_client
        self.config = config or {}
        
        # Configuration parameters
        self.similarity_threshold = self.config.get("similarity_threshold", 0.85)
        self.consolidation_enabled = self.config.get("consolidation_enabled", True)
        self.graph_pruning_enabled = self.config.get("graph_pruning_enabled", True)
        self.parallel_processing = self.config.get("parallel_processing", True)
        self.max_workers = self.config.get("max_workers", 4)
        
        # Pruning thresholds by tier
        self.tier_schedules = {
            "hourly": {"interval": timedelta(hours=1), "last_run": None},
            "daily": {"interval": timedelta(days=1), "last_run": None},
            "weekly": {"interval": timedelta(weeks=1), "last_run": None},
            "monthly": {"interval": timedelta(days=30), "last_run": None}
        }
        
        # Statistics tracking
        self.pruning_history: List[PruningStats] = []
        
        logger.info("Memory pruning engine initialized")
        
    async def run_tiered_pruning(
        self,
        user_id: Optional[str] = None,
        force_tiers: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> Dict[str, PruningStats]:
        """
        Run tiered pruning based on schedules or forced execution.
        
        Args:
            user_id: User ID for scoped pruning
            force_tiers: Force specific tiers to run regardless of schedule
            dry_run: Perform analysis without actual deletion
            
        Returns:
            Dictionary mapping tier names to pruning statistics
        """
        logger.info(f"Starting tiered pruning (dry_run={dry_run})")
        start_time = datetime.utcnow()
        
        tiers_to_run = force_tiers or self._get_scheduled_tiers()
        results = {}
        
        if not tiers_to_run:
            logger.info("No tiers scheduled for pruning")
            return results
            
        for tier in tiers_to_run:
            try:
                logger.info(f"Running {tier} pruning")
                stats = await self._run_tier_pruning(tier, user_id, dry_run)
                results[tier] = stats
                
                # Update last run time
                if not dry_run:
                    self.tier_schedules[tier]["last_run"] = datetime.utcnow()
                    
            except Exception as e:
                logger.error(f"Failed to run {tier} pruning: {e}")
                results[tier] = PruningStats(
                    total_memories_before=0,
                    total_memories_after=0,
                    memories_deleted=0,
                    memories_consolidated=0,
                    categories_processed=0,
                    processing_time_seconds=0.0,
                    errors=[str(e)]
                )
                
        total_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Tiered pruning completed in {total_time:.2f} seconds")
        
        return results
        
    def _get_scheduled_tiers(self) -> List[str]:
        """Get tiers that are due for pruning based on schedule."""
        now = datetime.utcnow()
        scheduled_tiers = []
        
        for tier, schedule in self.tier_schedules.items():
            last_run = schedule["last_run"]
            interval = schedule["interval"]
            
            if last_run is None or (now - last_run) >= interval:
                scheduled_tiers.append(tier)
                
        return scheduled_tiers
        
    async def _run_tier_pruning(
        self,
        tier: str,
        user_id: Optional[str],
        dry_run: bool
    ) -> PruningStats:
        """Run pruning for a specific tier."""
        start_time = datetime.utcnow()
        
        # Get categories and thresholds for this tier
        categories = MemoryCategoryConfig.get_categories_by_tier(tier)
        confidence_threshold = MemoryCategoryConfig.get_tier_threshold(tier)
        max_memories = MemoryCategoryConfig.get_tier_max_memories(tier)
        
        total_before = 0
        total_deleted = 0
        total_consolidated = 0
        errors = []
        
        # Process each category
        for category in categories:
            try:
                # Get memories for this category
                memories = await self.client.get_memories_by_category(
                    category, user_id, limit=max_memories, include_low_confidence=True
                )
                
                category_before = len(memories)
                total_before += category_before
                
                if category_before == 0:
                    continue
                    
                logger.info(f"Processing {category_before} memories for category {category.value}")
                
                # Apply pruning strategies
                pruning_results = await self._apply_pruning_strategies(
                    memories, category, confidence_threshold, dry_run
                )
                
                total_deleted += pruning_results["deleted"]
                total_consolidated += pruning_results["consolidated"]
                errors.extend(pruning_results["errors"])
                
            except Exception as e:
                logger.error(f"Failed to process category {category.value}: {e}")
                errors.append(f"Category {category.value}: {str(e)}")
                
        # Calculate final statistics
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        total_after = total_before - total_deleted
        
        stats = PruningStats(
            total_memories_before=total_before,
            total_memories_after=total_after,
            memories_deleted=total_deleted,
            memories_consolidated=total_consolidated,
            categories_processed=len(categories),
            processing_time_seconds=processing_time,
            errors=errors
        )
        
        # Track statistics
        if not dry_run:
            self.pruning_history.append(stats)
            
        logger.info(
            f"Tier {tier} pruning completed: "
            f"{total_deleted} deleted, {total_consolidated} consolidated, "
            f"{len(errors)} errors in {processing_time:.2f}s"
        )
        
        return stats
        
    async def _apply_pruning_strategies(
        self,
        memories: List[Dict[str, Any]],
        category: MemoryCategory,
        confidence_threshold: float,
        dry_run: bool
    ) -> Dict[str, int]:
        """Apply various pruning strategies to memories."""
        results = {"deleted": 0, "consolidated": 0, "errors": []}
        
        if not memories:
            return results
            
        try:
            # Strategy 1: Remove low-confidence memories
            low_confidence_results = await self._prune_low_confidence(
                memories, category, confidence_threshold, dry_run
            )
            results["deleted"] += low_confidence_results["deleted"]
            results["errors"].extend(low_confidence_results["errors"])
            
            # Get remaining memories after confidence pruning
            remaining_memories = [m for m in memories 
                                if not self._should_delete_low_confidence(m, category, confidence_threshold)]
            
            # Strategy 2: Remove aged-out memories
            aging_results = await self._prune_aged_memories(
                remaining_memories, category, dry_run
            )
            results["deleted"] += aging_results["deleted"]
            results["errors"].extend(aging_results["errors"])
            
            # Get memories after aging pruning
            remaining_memories = [m for m in remaining_memories 
                                if not self._should_delete_aged(m, category)]
            
            # Strategy 3: Consolidate similar memories
            if self.consolidation_enabled:
                consolidation_results = await self._consolidate_similar_memories(
                    remaining_memories, category, dry_run
                )
                results["consolidated"] += consolidation_results["consolidated"]
                results["errors"].extend(consolidation_results["errors"])
                
            # Strategy 4: Graph-based pruning
            if self.graph_pruning_enabled and MemoryCategoryConfig.requires_graph_pruning(category):
                graph_results = await self._prune_graph_memories(
                    remaining_memories, category, dry_run
                )
                results["deleted"] += graph_results["deleted"]
                results["errors"].extend(graph_results["errors"])
                
        except Exception as e:
            logger.error(f"Error applying pruning strategies: {e}")
            results["errors"].append(str(e))
            
        return results
        
    async def _prune_low_confidence(
        self,
        memories: List[Dict[str, Any]],
        category: MemoryCategory,
        threshold: float,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Remove memories below confidence threshold."""
        results = {"deleted": 0, "errors": []}
        
        memories_to_delete = [
            m for m in memories 
            if self._should_delete_low_confidence(m, category, threshold)
        ]
        
        if not memories_to_delete:
            return results
            
        logger.info(f"Pruning {len(memories_to_delete)} low-confidence memories")
        
        if not dry_run:
            memory_ids = [m["id"] for m in memories_to_delete]
            deletion_results = await self.client.batch_delete_memories(memory_ids)
            
            results["deleted"] = sum(1 for success in deletion_results.values() if success)
            failed_deletions = sum(1 for success in deletion_results.values() if not success)
            
            if failed_deletions > 0:
                results["errors"].append(f"Failed to delete {failed_deletions} low-confidence memories")
        else:
            results["deleted"] = len(memories_to_delete)
            
        return results
        
    def _should_delete_low_confidence(
        self,
        memory: Dict[str, Any],
        category: MemoryCategory,
        threshold: float
    ) -> bool:
        """Check if memory should be deleted due to low confidence."""
        metadata = memory.get("metadata", {})
        confidence = metadata.get("confidence", 0.5)
        
        # Apply category-specific threshold
        policy = MemoryCategoryConfig.get_policy(category)
        category_threshold = max(threshold, policy.min_confidence_threshold)
        
        # Consider access count and importance score
        access_count = metadata.get("access_count", 0)
        importance_score = metadata.get("importance_score", confidence)
        
        # Boost threshold for frequently accessed memories
        if access_count > 10:
            category_threshold -= 0.1
        elif access_count > 5:
            category_threshold -= 0.05
            
        return importance_score < category_threshold
        
    async def _prune_aged_memories(
        self,
        memories: List[Dict[str, Any]],
        category: MemoryCategory,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Remove memories that have exceeded their retention period."""
        results = {"deleted": 0, "errors": []}
        
        memories_to_delete = [
            m for m in memories 
            if self._should_delete_aged(m, category)
        ]
        
        if not memories_to_delete:
            return results
            
        logger.info(f"Pruning {len(memories_to_delete)} aged memories")
        
        if not dry_run:
            memory_ids = [m["id"] for m in memories_to_delete]
            deletion_results = await self.client.batch_delete_memories(memory_ids)
            
            results["deleted"] = sum(1 for success in deletion_results.values() if success)
            failed_deletions = sum(1 for success in deletion_results.values() if not success)
            
            if failed_deletions > 0:
                results["errors"].append(f"Failed to delete {failed_deletions} aged memories")
        else:
            results["deleted"] = len(memories_to_delete)
            
        return results
        
    def _should_delete_aged(self, memory: Dict[str, Any], category: MemoryCategory) -> bool:
        """Check if memory should be deleted due to age."""
        metadata = memory.get("metadata", {})
        created_at_str = metadata.get("created_at")
        
        if not created_at_str:
            return False
            
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            created_at = created_at.replace(tzinfo=None)
            age = datetime.utcnow() - created_at
            
            policy = MemoryCategoryConfig.get_policy(category)
            
            # Apply decay factor
            age_hours = age.total_seconds() / 3600
            decay_factor = MemoryCategoryConfig.calculate_decay_factor(category, age_hours)
            
            # Adjust confidence based on decay
            original_confidence = metadata.get("confidence", 0.5)
            decayed_confidence = original_confidence * decay_factor
            
            # Check if decayed confidence is below threshold or max age exceeded
            return (decayed_confidence < policy.min_confidence_threshold or 
                   age > policy.max_age)
                   
        except Exception as e:
            logger.error(f"Error calculating memory age: {e}")
            return False
            
    async def _consolidate_similar_memories(
        self,
        memories: List[Dict[str, Any]],
        category: MemoryCategory,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Consolidate similar memories to reduce redundancy."""
        results = {"consolidated": 0, "errors": []}
        
        if len(memories) < 2:
            return results
            
        try:
            # Find consolidation candidates
            candidates = await self._find_consolidation_candidates(memories, category)
            
            if not candidates:
                return results
                
            logger.info(f"Found {len(candidates)} consolidation candidates")
            
            for candidate in candidates:
                try:
                    if not dry_run:
                        success = await self._perform_consolidation(candidate, category)
                        if success:
                            results["consolidated"] += len(candidate.memories) - 1  # -1 because we create one new memory
                        else:
                            results["errors"].append("Failed to consolidate memory group")
                    else:
                        results["consolidated"] += len(candidate.memories) - 1
                        
                except Exception as e:
                    logger.error(f"Error consolidating memory group: {e}")
                    results["errors"].append(str(e))
                    
        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}")
            results["errors"].append(str(e))
            
        return results
        
    async def _find_consolidation_candidates(
        self,
        memories: List[Dict[str, Any]],
        category: MemoryCategory
    ) -> List[ConsolidationCandidate]:
        """Find groups of similar memories that can be consolidated."""
        candidates = []
        
        # Group memories by content similarity
        similarity_groups = defaultdict(list)
        processed_ids = set()
        
        for i, memory1 in enumerate(memories):
            if memory1["id"] in processed_ids:
                continue
                
            similar_memories = [memory1]
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if memory2["id"] in processed_ids:
                    continue
                    
                similarity = self._calculate_similarity(memory1, memory2)
                
                if similarity >= self.similarity_threshold:
                    similar_memories.append(memory2)
                    processed_ids.add(memory2["id"])
                    
            if len(similar_memories) >= MemoryCategoryConfig.get_policy(category).consolidation_threshold:
                processed_ids.add(memory1["id"])
                
                # Create consolidation candidate
                candidate = self._create_consolidation_candidate(similar_memories, category)
                candidates.append(candidate)
                
        return candidates
        
    def _calculate_similarity(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> float:
        """Calculate similarity score between two memories."""
        content1 = memory1.get("text", memory1.get("content", "")).lower()
        content2 = memory2.get("text", memory2.get("content", "")).lower()
        
        # Simple Jaccard similarity based on words
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Consider metadata similarity
        metadata1 = memory1.get("metadata", {})
        metadata2 = memory2.get("metadata", {})
        
        metadata_similarity = self._calculate_metadata_similarity(metadata1, metadata2)
        
        # Combined score (weighted average)
        return 0.7 * jaccard_similarity + 0.3 * metadata_similarity
        
    def _calculate_metadata_similarity(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> float:
        """Calculate similarity between metadata."""
        similarity_score = 0.0
        total_weight = 0.0
        
        # Compare important fields
        fields_weights = {
            "category": 0.3,
            "source": 0.2,
            "tags": 0.3,
            "confidence": 0.2
        }
        
        for field, weight in fields_weights.items():
            total_weight += weight
            
            value1 = metadata1.get(field)
            value2 = metadata2.get(field)
            
            if value1 == value2:
                similarity_score += weight
            elif field == "tags" and isinstance(value1, list) and isinstance(value2, list):
                # Calculate tag overlap
                tags1 = set(value1)
                tags2 = set(value2)
                if tags1 and tags2:
                    overlap = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
                    similarity_score += weight * overlap
            elif field == "confidence" and isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                # Calculate confidence similarity
                diff = abs(value1 - value2)
                confidence_similarity = max(0, 1 - diff)
                similarity_score += weight * confidence_similarity
                
        return similarity_score / total_weight if total_weight > 0 else 0.0
        
    def _create_consolidation_candidate(
        self,
        similar_memories: List[Dict[str, Any]],
        category: MemoryCategory
    ) -> ConsolidationCandidate:
        """Create a consolidation candidate from similar memories."""
        # Calculate average similarity
        total_similarity = 0.0
        comparison_count = 0
        
        for i in range(len(similar_memories)):
            for j in range(i+1, len(similar_memories)):
                total_similarity += self._calculate_similarity(similar_memories[i], similar_memories[j])
                comparison_count += 1
                
        avg_similarity = total_similarity / comparison_count if comparison_count > 0 else 1.0
        
        # Consolidate content
        contents = []
        confidences = []
        access_counts = []
        importance_scores = []
        all_tags = set()
        
        for memory in similar_memories:
            content = memory.get("text", memory.get("content", ""))
            if content and content not in contents:
                contents.append(content)
                
            metadata = memory.get("metadata", {})
            confidences.append(metadata.get("confidence", 0.5))
            access_counts.append(metadata.get("access_count", 0))
            importance_scores.append(metadata.get("importance_score", metadata.get("confidence", 0.5)))
            
            tags = metadata.get("tags", [])
            if isinstance(tags, list):
                all_tags.update(tags)
                
        # Create consolidated content
        if len(contents) == 1:
            consolidated_content = contents[0]
        else:
            consolidated_content = " | ".join(contents[:3])  # Limit to top 3 variations
            
        # Calculate combined confidence (weighted by importance)
        if importance_scores:
            weights = [score for score in importance_scores]
            total_weight = sum(weights)
            combined_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
        else:
            combined_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
        # Create combined metadata
        combined_metadata = {
            "category": category.value,
            "confidence": combined_confidence,
            "source": "consolidation",
            "consolidated_from": len(similar_memories),
            "total_access_count": sum(access_counts),
            "importance_score": combined_confidence + 0.1,  # Boost for consolidation
            "tags": list(all_tags),
            "created_at": datetime.utcnow().isoformat()
        }
        
        return ConsolidationCandidate(
            memories=similar_memories,
            similarity_score=avg_similarity,
            consolidated_content=consolidated_content,
            combined_confidence=combined_confidence,
            combined_metadata=combined_metadata
        )
        
    async def _perform_consolidation(
        self,
        candidate: ConsolidationCandidate,
        category: MemoryCategory
    ) -> bool:
        """Perform the actual consolidation of memories."""
        try:
            # Create consolidated memory
            await self.client.add_memory(
                content=candidate.consolidated_content,
                category=category,
                confidence=candidate.combined_confidence,
                metadata=candidate.combined_metadata
            )
            
            # Delete original memories
            memory_ids = [m["id"] for m in candidate.memories]
            deletion_results = await self.client.batch_delete_memories(memory_ids)
            
            # Check if all deletions were successful
            success_count = sum(1 for success in deletion_results.values() if success)
            
            if success_count != len(memory_ids):
                logger.warning(f"Only {success_count}/{len(memory_ids)} memories deleted during consolidation")
                
            logger.info(f"Consolidated {len(candidate.memories)} memories into 1")
            return True
            
        except Exception as e:
            logger.error(f"Failed to perform consolidation: {e}")
            return False
            
    async def _prune_graph_memories(
        self,
        memories: List[Dict[str, Any]],
        category: MemoryCategory,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Prune graph-based memories using relationship analysis."""
        results = {"deleted": 0, "errors": []}
        
        if not self.graph_pruning_enabled:
            return results
            
        try:
            # Find orphaned nodes (memories with broken relationships)
            orphaned_memories = []
            
            for memory in memories:
                metadata = memory.get("metadata", {})
                relationships = metadata.get("relationships", [])
                
                if relationships and await self._has_broken_relationships(memory, relationships):
                    orphaned_memories.append(memory)
                    
            if orphaned_memories:
                logger.info(f"Found {len(orphaned_memories)} orphaned graph memories")
                
                if not dry_run:
                    memory_ids = [m["id"] for m in orphaned_memories]
                    deletion_results = await self.client.batch_delete_memories(memory_ids)
                    results["deleted"] = sum(1 for success in deletion_results.values() if success)
                else:
                    results["deleted"] = len(orphaned_memories)
                    
        except Exception as e:
            logger.error(f"Error in graph memory pruning: {e}")
            results["errors"].append(str(e))
            
        return results
        
    async def _has_broken_relationships(
        self,
        memory: Dict[str, Any],
        relationships: List[str]
    ) -> bool:
        """Check if a memory has broken relationships."""
        try:
            # Check if related memories still exist
            existing_count = 0
            
            for related_id in relationships:
                # Try to find the related memory
                search_results = await self.client.search_memories(
                    query="*",
                    filters={"id": related_id},
                    limit=1
                )
                
                if search_results:
                    existing_count += 1
                    
            # Consider relationships broken if more than 50% are missing
            broken_threshold = len(relationships) * 0.5
            return existing_count < broken_threshold
            
        except Exception as e:
            logger.error(f"Error checking relationships: {e}")
            return False
            
    async def optimize_memory_storage(
        self,
        user_id: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Optimize memory storage by defragmenting and reorganizing."""
        logger.info("Starting memory storage optimization")
        start_time = datetime.utcnow()
        
        results = {
            "memories_optimized": 0,
            "categories_processed": 0,
            "storage_savings": 0,
            "errors": []
        }
        
        try:
            # Get memory statistics
            stats = await self.client.get_memory_stats(user_id)
            
            # Process each category
            for category_str, count in stats["by_category"].items():
                if count == 0:
                    continue
                    
                try:
                    category = MemoryCategory(category_str)
                    memories = await self.client.get_memories_by_category(category, user_id)
                    
                    # Optimize memory metadata
                    optimization_results = await self._optimize_category_memories(
                        memories, category, dry_run
                    )
                    
                    results["memories_optimized"] += optimization_results["optimized"]
                    results["storage_savings"] += optimization_results["savings"]
                    results["errors"].extend(optimization_results["errors"])
                    results["categories_processed"] += 1
                    
                except ValueError:
                    # Skip unknown categories
                    continue
                except Exception as e:
                    logger.error(f"Error optimizing category {category_str}: {e}")
                    results["errors"].append(f"Category {category_str}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error during memory optimization: {e}")
            results["errors"].append(str(e))
            
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        results["processing_time_seconds"] = processing_time
        
        logger.info(f"Memory optimization completed in {processing_time:.2f} seconds")
        return results
        
    async def _optimize_category_memories(
        self,
        memories: List[Dict[str, Any]],
        category: MemoryCategory,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Optimize memories for a specific category."""
        results = {"optimized": 0, "savings": 0, "errors": []}
        
        for memory in memories:
            try:
                original_size = len(json.dumps(memory).encode('utf-8'))
                
                # Optimize metadata
                optimized_metadata = self._optimize_metadata(memory.get("metadata", {}), category)
                
                # Calculate size savings
                optimized_memory = memory.copy()
                optimized_memory["metadata"] = optimized_metadata
                optimized_size = len(json.dumps(optimized_memory).encode('utf-8'))
                
                savings = original_size - optimized_size
                
                if savings > 0 and not dry_run:
                    # Update memory with optimized metadata
                    await self.client.update_memory(
                        memory_id=memory["id"],
                        metadata=optimized_metadata
                    )
                    
                    results["optimized"] += 1
                    results["savings"] += savings
                    
                elif savings > 0:
                    results["optimized"] += 1
                    results["savings"] += savings
                    
            except Exception as e:
                logger.error(f"Error optimizing memory {memory.get('id', 'unknown')}: {e}")
                results["errors"].append(str(e))
                
        return results
        
    def _optimize_metadata(self, metadata: Dict[str, Any], category: MemoryCategory) -> Dict[str, Any]:
        """Optimize memory metadata by removing redundant fields and compacting data."""
        optimized = metadata.copy()
        
        # Remove empty or None values
        optimized = {k: v for k, v in optimized.items() if v is not None and v != ""}
        
        # Compress tags (remove duplicates, limit length)
        if "tags" in optimized and isinstance(optimized["tags"], list):
            tags = list(set(optimized["tags"]))  # Remove duplicates
            optimized["tags"] = tags[:10]  # Limit to 10 tags
            
        # Round confidence to 2 decimal places
        if "confidence" in optimized and isinstance(optimized["confidence"], float):
            optimized["confidence"] = round(optimized["confidence"], 2)
            
        # Remove redundant timestamps (keep only most recent)
        timestamp_fields = ["created_at", "last_accessed", "updated_at"]
        timestamps = {field: optimized.get(field) for field in timestamp_fields if field in optimized}
        
        if timestamps:
            # Keep only the most recent timestamp and created_at
            if "created_at" in timestamps:
                optimized["created_at"] = timestamps["created_at"]
                
            # Keep the most recent of the other timestamps
            other_timestamps = {k: v for k, v in timestamps.items() if k != "created_at" and v}
            if other_timestamps:
                most_recent_field = max(other_timestamps.keys(), key=lambda k: other_timestamps[k])
                optimized["last_update"] = other_timestamps[most_recent_field]
                
            # Remove individual timestamp fields except created_at
            for field in timestamp_fields:
                if field != "created_at" and field in optimized:
                    del optimized[field]
                    
        return optimized
        
    async def get_pruning_recommendations(
        self,
        user_id: Optional[str] = None,
        category: Optional[MemoryCategory] = None
    ) -> Dict[str, Any]:
        """Get recommendations for memory pruning without performing actual pruning."""
        logger.info("Generating pruning recommendations")
        
        recommendations = {
            "total_memories": 0,
            "recommended_deletions": 0,
            "recommended_consolidations": 0,
            "categories": {},
            "storage_impact": {"current_mb": 0, "after_mb": 0, "savings_mb": 0},
            "confidence_analysis": {},
            "age_analysis": {}
        }
        
        try:
            # Get memory statistics
            stats = await self.client.get_memory_stats(user_id)
            recommendations["total_memories"] = stats["total_memories"]
            recommendations["confidence_analysis"] = stats["confidence_distribution"]
            recommendations["age_analysis"] = stats["age_distribution"]
            
            # Analyze each category
            categories_to_analyze = [category] if category else list(MemoryCategory)
            
            for cat in categories_to_analyze:
                if cat.value not in stats["by_category"]:
                    continue
                    
                memories = await self.client.get_memories_by_category(cat, user_id, include_low_confidence=True)
                
                if not memories:
                    continue
                    
                # Analyze category
                category_analysis = await self._analyze_category_for_recommendations(memories, cat)
                recommendations["categories"][cat.value] = category_analysis
                
                recommendations["recommended_deletions"] += category_analysis["recommended_deletions"]
                recommendations["recommended_consolidations"] += category_analysis["recommended_consolidations"]
                
        except Exception as e:
            logger.error(f"Error generating pruning recommendations: {e}")
            recommendations["error"] = str(e)
            
        return recommendations
        
    async def _analyze_category_for_recommendations(
        self,
        memories: List[Dict[str, Any]],
        category: MemoryCategory
    ) -> Dict[str, Any]:
        """Analyze a category and provide pruning recommendations."""
        analysis = {
            "total_memories": len(memories),
            "recommended_deletions": 0,
            "recommended_consolidations": 0,
            "low_confidence_count": 0,
            "aged_count": 0,
            "duplicate_groups": 0,
            "storage_mb": 0
        }
        
        if not memories:
            return analysis
            
        # Calculate storage size
        total_size = sum(len(json.dumps(memory).encode('utf-8')) for memory in memories)
        analysis["storage_mb"] = total_size / (1024 * 1024)
        
        policy = MemoryCategoryConfig.get_policy(category)
        
        # Analyze each memory
        for memory in memories:
            # Check for low confidence
            if self._should_delete_low_confidence(memory, category, policy.min_confidence_threshold):
                analysis["low_confidence_count"] += 1
                
            # Check for aging
            if self._should_delete_aged(memory, category):
                analysis["aged_count"] += 1
                
        # Find potential consolidations
        if self.consolidation_enabled:
            candidates = await self._find_consolidation_candidates(memories, category)
            analysis["duplicate_groups"] = len(candidates)
            
            for candidate in candidates:
                analysis["recommended_consolidations"] += len(candidate.memories) - 1
                
        # Calculate total recommended deletions
        analysis["recommended_deletions"] = analysis["low_confidence_count"] + analysis["aged_count"]
        
        return analysis
        
    def get_pruning_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent pruning history."""
        recent_history = self.pruning_history[-limit:] if self.pruning_history else []
        
        return [
            {
                "timestamp": datetime.utcnow().isoformat(),  # In real implementation, store actual timestamp
                "total_memories_before": stats.total_memories_before,
                "total_memories_after": stats.total_memories_after,
                "memories_deleted": stats.memories_deleted,
                "memories_consolidated": stats.memories_consolidated,
                "categories_processed": stats.categories_processed,
                "processing_time_seconds": stats.processing_time_seconds,
                "deletion_rate": stats.deletion_rate,
                "consolidation_rate": stats.consolidation_rate,
                "error_count": len(stats.errors)
            }
            for stats in recent_history
        ]