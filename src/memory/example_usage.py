"""
Example usage of the sophisticated memory pruning engine for the Aerodrome AI Agent.

This script demonstrates how to use the memory system with various pruning strategies,
batch operations, and graph memory features.
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

from memory_categories import MemoryCategory, MemoryCategoryConfig, MemoryMetadata
from mem0_client import EnhancedMem0Client
from pruning_engine import MemoryPruningEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_setup_client():
    """Example of setting up the enhanced Mem0 client."""
    # Configuration
    config = {
        "batch_size": 100,
        "max_retries": 3,
        "retry_delay": 1.0
    }
    
    # Neo4j configuration for graph memory
    neo4j_config = {
        "url": os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j")
    }
    
    # Initialize client
    client = EnhancedMem0Client(
        api_key=os.getenv("MEM0_API_KEY"),
        host=os.getenv("MEM0_HOST", "https://api.mem0.ai"),
        version="v2",
        config=config,
        enable_graph=True,
        neo4j_config=neo4j_config
    )
    
    return client


async def example_add_memories(client: EnhancedMem0Client):
    """Example of adding memories to different categories."""
    logger.info("Adding example memories...")
    
    # Example memories for different categories
    example_memories = [
        {
            "content": "Aerodrome protocol uses ve(3,3) tokenomics model with vote-escrowed tokens",
            "category": MemoryCategory.PROTOCOL_CONSTANTS,
            "confidence": 0.95,
            "metadata": {"source": "documentation", "stability_score": 0.9}
        },
        {
            "content": "ETH-USDC pool on Base showing 15.2% APY with $2.3M TVL",
            "category": MemoryCategory.POOL_PERFORMANCE,
            "confidence": 0.8,
            "metadata": {"pool_tvl": 2300000, "apy": 15.2, "chain": "base"}
        },
        {
            "content": "Large holders consistently vote for high-emission pools",
            "category": MemoryCategory.VOTING_PATTERNS,
            "confidence": 0.75,
            "metadata": {"vote_weight": 150000, "pattern_strength": 0.8}
        },
        {
            "content": "Strong correlation (0.82) between AERO price and Base TVL growth",
            "category": MemoryCategory.MARKET_CORRELATIONS,
            "confidence": 0.85,
            "metadata": {"correlation_strength": 0.82, "timeframe": "30d"}
        },
        {
            "content": "Speculation: New partnership announcement might boost AERO price",
            "category": MemoryCategory.SPECULATIVE_INSIGHTS,
            "confidence": 0.4,
            "metadata": {"speculation_level": 0.9, "source": "social_media"}
        }
    ]
    
    # Add memories individually
    for memory_data in example_memories:
        try:
            result = await client.add_memory(
                content=memory_data["content"],
                category=memory_data["category"],
                confidence=memory_data["confidence"],
                metadata=memory_data["metadata"]
            )
            logger.info(f"Added memory: {result.get('id', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
    
    # Batch add memories
    batch_memories = [
        {
            "content": "Pool performance declined 5% over last week",
            "confidence": 0.7,
            "metadata": {"performance_change": -0.05, "timeframe": "1w"}
        },
        {
            "content": "New liquidity incentives program launched",
            "confidence": 0.9,
            "metadata": {"program_type": "incentives", "status": "active"}
        }
    ]
    
    try:
        batch_results = await client.batch_add_memories(
            memories=batch_memories,
            category=MemoryCategory.POOL_PERFORMANCE,
            user_id="aerodrome_agent"
        )
        logger.info(f"Batch added {len(batch_results)} memories")
    except Exception as e:
        logger.error(f"Batch add failed: {e}")


async def example_search_memories(client: EnhancedMem0Client):
    """Example of searching memories with advanced filters."""
    logger.info("Searching memories with different filters...")
    
    # Basic search
    results = await client.search_memories(
        query="pool performance",
        limit=5
    )
    logger.info(f"Basic search returned {len(results)} results")
    
    # Category-specific search
    results = await client.search_memories(
        query="AERO token",
        category=MemoryCategory.PROTOCOL_CONSTANTS,
        confidence_threshold=0.8,
        limit=10
    )
    logger.info(f"Category search returned {len(results)} results")
    
    # Advanced filtering
    results = await client.search_memories(
        query="correlation",
        filters={
            "correlation_strength": {"$gte": 0.7},
            "timeframe": {"$in": ["30d", "7d"]}
        },
        limit=5
    )
    logger.info(f"Advanced filter search returned {len(results)} results")
    
    # Get all memories by category
    pool_memories = await client.get_memories_by_category(
        category=MemoryCategory.POOL_PERFORMANCE,
        limit=100
    )
    logger.info(f"Found {len(pool_memories)} pool performance memories")


async def example_pruning_engine(client: EnhancedMem0Client):
    """Example of using the memory pruning engine."""
    logger.info("Initializing memory pruning engine...")
    
    # Engine configuration
    engine_config = {
        "similarity_threshold": 0.85,
        "consolidation_enabled": True,
        "graph_pruning_enabled": True,
        "parallel_processing": True,
        "max_workers": 4
    }
    
    # Initialize pruning engine
    pruning_engine = MemoryPruningEngine(
        mem0_client=client,
        config=engine_config
    )
    
    # Get pruning recommendations (dry run)
    logger.info("Getting pruning recommendations...")
    recommendations = await pruning_engine.get_pruning_recommendations()
    
    logger.info(f"Pruning recommendations:")
    logger.info(f"  Total memories: {recommendations['total_memories']}")
    logger.info(f"  Recommended deletions: {recommendations['recommended_deletions']}")
    logger.info(f"  Recommended consolidations: {recommendations['recommended_consolidations']}")
    
    for category, analysis in recommendations["categories"].items():
        logger.info(f"  {category}: {analysis['total_memories']} memories, "
                   f"{analysis['recommended_deletions']} deletions, "
                   f"{analysis['recommended_consolidations']} consolidations")
    
    # Run tiered pruning (dry run first)
    logger.info("Running tiered pruning (dry run)...")
    dry_run_results = await pruning_engine.run_tiered_pruning(
        force_tiers=["hourly", "daily"],
        dry_run=True
    )
    
    for tier, stats in dry_run_results.items():
        logger.info(f"Dry run {tier} pruning:")
        logger.info(f"  Would delete: {stats.memories_deleted} memories")
        logger.info(f"  Would consolidate: {stats.memories_consolidated} memories")
        logger.info(f"  Processing time: {stats.processing_time_seconds:.2f}s")
    
    # Actual pruning (uncomment to run)
    # logger.info("Running actual tiered pruning...")
    # actual_results = await pruning_engine.run_tiered_pruning(
    #     force_tiers=["hourly"],
    #     dry_run=False
    # )
    # 
    # for tier, stats in actual_results.items():
    #     logger.info(f"Actual {tier} pruning:")
    #     logger.info(f"  Deleted: {stats.memories_deleted} memories")
    #     logger.info(f"  Consolidated: {stats.memories_consolidated} memories")
    #     logger.info(f"  Deletion rate: {stats.deletion_rate:.1f}%")


async def example_memory_optimization(client: EnhancedMem0Client):
    """Example of memory storage optimization."""
    logger.info("Running memory storage optimization...")
    
    pruning_engine = MemoryPruningEngine(client)
    
    # Optimize memory storage (dry run)
    optimization_results = await pruning_engine.optimize_memory_storage(
        dry_run=True
    )
    
    logger.info(f"Storage optimization results:")
    logger.info(f"  Memories optimized: {optimization_results['memories_optimized']}")
    logger.info(f"  Categories processed: {optimization_results['categories_processed']}")
    logger.info(f"  Storage savings: {optimization_results['storage_savings']} bytes")
    logger.info(f"  Processing time: {optimization_results['processing_time_seconds']:.2f}s")
    
    if optimization_results["errors"]:
        logger.warning(f"Optimization errors: {len(optimization_results['errors'])}")


async def example_export_import(client: EnhancedMem0Client):
    """Example of memory export and import operations."""
    logger.info("Exporting memories...")
    
    # Export all memories as JSON
    all_memories_json = await client.export_memories(
        format_type="json",
        include_metadata=True
    )
    logger.info(f"Exported {all_memories_json['memory_count']} memories to JSON")
    
    # Export specific category as CSV
    pool_memories_csv = await client.export_memories(
        category=MemoryCategory.POOL_PERFORMANCE,
        format_type="csv",
        include_metadata=True
    )
    logger.info(f"Exported pool performance memories to CSV")
    
    # Example import (uncomment to test)
    # logger.info("Importing memories...")
    # import_results = await client.import_memories(
    #     data=all_memories_json,
    #     format_type="json",
    #     default_category=MemoryCategory.SPECULATIVE_INSIGHTS
    # )
    # logger.info(f"Import results: {import_results['success']} success, {import_results['failed']} failed")


async def example_memory_stats(client: EnhancedMem0Client):
    """Example of getting comprehensive memory statistics."""
    logger.info("Getting memory statistics...")
    
    stats = await client.get_memory_stats()
    
    logger.info(f"Memory Statistics:")
    logger.info(f"  Total memories: {stats['total_memories']}")
    logger.info(f"  Graph memories: {stats['graph_memories']}")
    
    logger.info(f"  By category:")
    for category, count in stats["by_category"].items():
        logger.info(f"    {category}: {count}")
    
    logger.info(f"  Confidence distribution:")
    for level, count in stats["confidence_distribution"].items():
        logger.info(f"    {level}: {count}")
    
    logger.info(f"  Age distribution:")
    for age_group, count in stats["age_distribution"].items():
        logger.info(f"    {age_group}: {count}")


async def example_category_configuration():
    """Example of working with memory categories and policies."""
    logger.info("Memory category configuration examples...")
    
    # Get policy for a specific category
    policy = MemoryCategoryConfig.get_policy(MemoryCategory.POOL_PERFORMANCE)
    logger.info(f"Pool performance policy:")
    logger.info(f"  Max age: {policy.max_age}")
    logger.info(f"  Min confidence: {policy.min_confidence_threshold}")
    logger.info(f"  Decay rate: {policy.decay_rate}")
    logger.info(f"  Batch size: {policy.batch_size}")
    
    # Calculate decay factor
    decay_factor = MemoryCategoryConfig.calculate_decay_factor(
        MemoryCategory.MARKET_CORRELATIONS,
        age_hours=72  # 3 days
    )
    logger.info(f"Decay factor for 3-day-old market correlation: {decay_factor:.3f}")
    
    # Check consolidation threshold
    should_consolidate = MemoryCategoryConfig.should_consolidate(
        MemoryCategory.SPECULATIVE_INSIGHTS,
        similar_count=10
    )
    logger.info(f"Should consolidate 10 similar speculative insights: {should_consolidate}")
    
    # Get custom filters
    custom_filters = MemoryCategoryConfig.get_custom_filters(MemoryCategory.VOTING_PATTERNS)
    logger.info(f"Custom filters for voting patterns: {custom_filters}")


async def main():
    """Main example function."""
    try:
        # Setup client
        logger.info("Setting up enhanced Mem0 client...")
        async with await example_setup_client() as client:
            
            # Run examples
            await example_category_configuration()
            await example_add_memories(client)
            await example_search_memories(client)
            await example_memory_stats(client)
            await example_pruning_engine(client)
            await example_memory_optimization(client)
            await example_export_import(client)
            
            logger.info("All examples completed successfully!")
            
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())