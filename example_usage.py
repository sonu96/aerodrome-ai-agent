"""
Example usage of the Aerodrome Brain Confidence Scoring System.

This script demonstrates how to use the confidence scoring system
for real-world Aerodrome protocol intelligence.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.brain import (
    ConfidenceScorer, 
    MemoryCategory, 
    MemoryItem, 
    DataSourceType,
    MetricsCollector,
    MetricsAnalyzer,
    MetricsReporter
)


async def example_pool_analysis():
    """Example: Analyze pool performance with confidence scoring."""
    print("=== Pool Performance Analysis Example ===\n")
    
    # Initialize the confidence scoring system
    scorer = ConfidenceScorer()
    collector = MetricsCollector()
    analyzer = MetricsAnalyzer(collector)
    reporter = MetricsReporter(analyzer)
    
    # Create memory items for different pool observations
    pool_memories = [
        MemoryItem(
            id="aerodrome_pool_usdc_eth",
            category=MemoryCategory.POOL_PERFORMANCE,
            data={
                "pool_address": "0x6cDcb1C4A4D1C3C6d054b27AC5B77e89eAFb971d",
                "pair": "USDC/ETH",
                "volume_24h": 2_500_000,
                "fees_24h": 7_500,
                "tvl": 15_000_000,
                "sample_size": 144  # 24 hours of 10-minute intervals
            },
            confidence=0.0,  # Will be calculated
            created_at=datetime.now() - timedelta(hours=4),
            updated_at=datetime.now(),
            source_type=DataSourceType.API_ENDPOINT,
            corroborating_sources={"quicknode", "dexscreener", "defillama"}
        ),
        MemoryItem(
            id="aerodrome_voting_analysis", 
            category=MemoryCategory.VOTING_PATTERNS,
            data={
                "epoch": 15,
                "total_votes": 1_200_000,
                "bribe_effectiveness": 0.85,
                "voter_participation": 0.72,
                "sample_size": 50  # 50 voting wallets analyzed
            },
            confidence=0.0,
            created_at=datetime.now() - timedelta(hours=1),
            updated_at=datetime.now(),
            source_type=DataSourceType.BLOCKCHAIN_DATA,
            corroborating_sources={"aerodrome_api", "snapshot"}
        ),
        MemoryItem(
            id="market_correlation_btc",
            category=MemoryCategory.MARKET_CORRELATIONS,
            data={
                "correlation_coefficient": 0.73,
                "timeframe": "7d",
                "p_value": 0.001,
                "sample_size": 168  # 7 days of hourly data
            },
            confidence=0.0,
            created_at=datetime.now() - timedelta(minutes=30),
            updated_at=datetime.now(),
            source_type=DataSourceType.PREDICTED_VALUE,
            corroborating_sources={"coingecko"}
        )
    ]
    
    # Calculate confidence scores for each memory
    for memory in pool_memories:
        confidence, factors = await scorer.calculate_confidence(
            memory,
            metadata={
                "provider": "quicknode" if memory.source_type == DataSourceType.API_ENDPOINT else "internal",
                "uptime": 0.999,
                "sample_size": memory.data.get("sample_size", 1)
            }
        )
        memory.confidence = confidence
        
        print(f"Memory: {memory.id}")
        print(f"  Category: {memory.category.value}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Factors:")
        print(f"    - Data reliability: {factors.data_source_reliability:.3f}")
        print(f"    - Historical accuracy: {factors.historical_accuracy:.3f}")
        print(f"    - Recency: {factors.recency:.3f}")
        print(f"    - Corroboration: {factors.corroboration:.3f}")
        print(f"    - Sample size: {factors.sample_size:.3f}")
        print(f"  Should retain: {scorer.should_retain_memory(memory)}")
        print()
    
    return pool_memories, scorer, collector, analyzer, reporter


async def example_prediction_validation():
    """Example: Validate predictions and update confidence scores."""
    print("=== Prediction Validation Example ===\n")
    
    pool_memories, scorer, collector, analyzer, reporter = await example_pool_analysis()
    
    # Simulate prediction validations
    validations = [
        {
            "memory_id": "aerodrome_pool_usdc_eth",
            "prediction_timestamp": datetime.now() - timedelta(hours=6),
            "predicted_volume": 2_500_000,
            "actual_volume": 2_350_000,  # 94% accuracy
            "category": MemoryCategory.POOL_PERFORMANCE
        },
        {
            "memory_id": "aerodrome_voting_analysis", 
            "prediction_timestamp": datetime.now() - timedelta(hours=2),
            "predicted_participation": 0.72,
            "actual_participation": 0.68,  # 94.4% accuracy
            "category": MemoryCategory.VOTING_PATTERNS
        },
        {
            "memory_id": "market_correlation_btc",
            "prediction_timestamp": datetime.now() - timedelta(hours=1),
            "predicted_correlation": 0.73,
            "actual_correlation": 0.68,  # 93.2% accuracy
            "category": MemoryCategory.MARKET_CORRELATIONS
        }
    ]
    
    # Record prediction validations
    for validation in validations:
        await collector.record_prediction_validation(
            memory_id=validation["memory_id"],
            prediction_timestamp=validation["prediction_timestamp"], 
            predicted_outcome=validation.get("predicted_volume") or validation.get("predicted_participation") or validation.get("predicted_correlation"),
            actual_outcome=validation.get("actual_volume") or validation.get("actual_participation") or validation.get("actual_correlation"),
            confidence_at_prediction=next(m.confidence for m in pool_memories if m.id == validation["memory_id"]),
            category=validation["category"]
        )
        
        # Update prediction outcome in scorer
        memory_item = next(m for m in pool_memories if m.id == validation["memory_id"])
        await scorer.update_prediction_outcome(validation["memory_id"], True, memory_item)
    
    print(f"Recorded {len(validations)} prediction validations")
    return pool_memories, scorer, collector, analyzer, reporter


async def example_metrics_analysis():
    """Example: Analyze confidence metrics and generate insights."""
    print("\n=== Metrics Analysis Example ===\n")
    
    pool_memories, scorer, collector, analyzer, reporter = await example_prediction_validation()
    
    # Add some accuracy metrics
    for memory in pool_memories:
        await collector.record_accuracy(
            memory.id,
            memory.category,
            memory.confidence,
            0.85  # Simulated actual accuracy
        )
    
    # Analyze trends
    print("Analyzing accuracy trends...")
    trends = await analyzer.analyze_accuracy_trends()
    print(json.dumps(trends, indent=2, default=str))
    
    # Analyze factor correlations
    print("\nAnalyzing factor correlations...")
    correlations = await analyzer.analyze_factor_correlations()
    for corr in correlations:
        print(f"- {corr.factor_name}: {corr.correlation_coefficient:.3f} (n={corr.sample_size})")
    
    # Generate comprehensive insights
    print("\nGenerating confidence insights...")
    insights = await analyzer.generate_confidence_insights()
    print("System Health:", insights.get("prediction_performance", {}).get("overall_accuracy", "Unknown"))
    print("Recommendations:")
    for rec in insights.get("recommendations", []):
        print(f"- {rec}")
    
    # Generate daily report
    print("\nGenerating daily report...")
    report = await reporter.generate_daily_report()
    print(f"System Health: {report['summary']['system_health']}")
    print(f"Metrics Collected: {report['summary']['total_metrics_collected']}")
    
    return insights


async def example_category_thresholds():
    """Example: Demonstrate category-specific thresholds and retention."""
    print("\n=== Category Thresholds Example ===\n")
    
    from src.brain.confidence_scorer import ConfidenceThresholds
    
    print("Confidence thresholds and retention periods by category:")
    print("-" * 60)
    
    for category in MemoryCategory:
        threshold = ConfidenceThresholds.get_threshold(category)
        retention = ConfidenceThresholds.get_retention_period(category)
        
        print(f"{category.value:25} | {threshold:>5.2f} | {retention.days:>3d} days")
    
    print("-" * 60)
    print("\nExample memory retention decisions:")
    
    # Test memories with different confidence levels
    test_memories = [
        ("Protocol constant", MemoryCategory.PROTOCOL_CONSTANTS, 0.98, "RETAIN"),
        ("Pool performance", MemoryCategory.POOL_PERFORMANCE, 0.65, "DISCARD"),
        ("Voting pattern", MemoryCategory.VOTING_PATTERNS, 0.75, "RETAIN"),
        ("Market correlation", MemoryCategory.MARKET_CORRELATIONS, 0.45, "DISCARD"),
        ("Speculative insight", MemoryCategory.SPECULATIVE_INSIGHTS, 0.35, "RETAIN")
    ]
    
    scorer = ConfidenceScorer()
    
    for name, category, confidence, expected in test_memories:
        memory = MemoryItem(
            id=f"test_{name.replace(' ', '_')}",
            category=category,
            data={},
            confidence=confidence,
            created_at=datetime.now() - timedelta(hours=2),
            updated_at=datetime.now(),
            source_type=DataSourceType.API_ENDPOINT
        )
        
        should_retain = scorer.should_retain_memory(memory)
        status = "RETAIN" if should_retain else "DISCARD"
        status_icon = "✓" if status == expected else "✗"
        
        print(f"{status_icon} {name:20} | {confidence:>5.2f} | {status:>7}")


async def main():
    """Run all examples."""
    print("Aerodrome Brain Confidence Scoring System - Examples\n")
    print("=" * 60)
    
    # Run examples
    await example_pool_analysis()
    await example_prediction_validation() 
    await example_metrics_analysis()
    await example_category_thresholds()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully! ✅")
    print("\nThe confidence scoring system is ready for production use.")


if __name__ == "__main__":
    asyncio.run(main())