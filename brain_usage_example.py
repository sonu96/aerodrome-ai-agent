#!/usr/bin/env python3
"""
Aerodrome Brain Usage Example

This example demonstrates how to use the Aerodrome Brain system for:
1. Initializing the brain with proper configuration
2. Processing various types of queries
3. Monitoring system health and performance
4. Handling different user scenarios

This serves as both documentation and a practical guide for implementation.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any

# Import the brain system
from src.brain import (
    AerodromeBrain,
    BrainConfig,
    QueryContext,
    SystemStatus
)


async def create_production_brain() -> AerodromeBrain:
    """
    Create a production-ready brain instance with proper configuration.
    
    This function shows how to configure the brain for real-world usage.
    """
    
    # Production configuration
    config = BrainConfig(
        # Memory system configuration
        memory_config={
            "api_key": os.getenv("MEM0_API_KEY"),  # Set in environment
            "enable_graph": True,  # Enable graph memory for better relationships
            "client_config": {
                "batch_size": 100,
                "timeout": 30
            },
            "neo4j_config": {  # Optional graph database config
                "url": os.getenv("NEO4J_URL", "bolt://localhost:7687"),
                "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "password")
            }
        },
        
        # Protocol integration configuration
        protocol_config={
            "quicknode_url": os.getenv("QUICKNODE_URL"),  # Your QuickNode endpoint
            "client_config": {
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 1.0
            },
            "monitor_config": {
                "update_interval": 300,  # 5 minutes
                "max_pools_to_monitor": 50
            },
            "voting_config": {
                "epoch_update_interval": 3600  # 1 hour
            }
        },
        
        # AI intelligence configuration
        intelligence_config={
            "gemini_api_key": os.getenv("GEMINI_API_KEY"),  # Set in environment
            "model": "gemini-2.0-flash-001",
            "client_config": {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 8192
            }
        },
        
        # System performance settings
        health_check_interval=60,  # 1 minute health checks
        metrics_update_interval=300,  # 5 minute metrics updates
        auto_pruning_interval=3600,  # 1 hour memory pruning
        max_concurrent_queries=10,  # Allow 10 concurrent queries
        
        # Safety and reliability settings
        max_memory_items=50000,  # Limit memory growth
        confidence_threshold=0.3,  # Minimum confidence for responses
        emergency_shutdown_threshold=50,  # Shutdown after 50 consecutive errors
        
        # Logging configuration
        log_level="INFO",
        enable_detailed_logging=True
    )
    
    # Create and initialize the brain
    brain = AerodromeBrain(config)
    await brain.initialize()
    
    return brain


async def demonstrate_query_types():
    """
    Demonstrate different types of queries the brain can handle.
    """
    print("🧠 Creating Aerodrome Brain...")
    
    # For demo purposes, use a simple configuration
    # In production, use create_production_brain() instead
    config = BrainConfig(
        memory_config={"api_key": "demo_key"},
        protocol_config={"quicknode_url": "demo_url"},
        intelligence_config={"gemini_api_key": "demo_key"},
        max_concurrent_queries=3
    )
    
    async with AerodromeBrain(config).lifespan() as brain:
        
        # Example queries demonstrating different capabilities
        example_queries = [
            {
                "query": "What is Aerodrome and how does it work?",
                "context": {"user_level": "beginner"},
                "description": "Educational query for beginners"
            },
            {
                "query": "What are the top performing pools by TVL today?",
                "context": {"user_preferences": {"format": "structured"}},
                "description": "Pool performance inquiry"
            },
            {
                "query": "How does veAERO voting work and when is the next epoch?",
                "context": {"user_level": "intermediate"},
                "description": "Voting system inquiry"
            },
            {
                "query": "Show me the current protocol metrics and health",
                "context": {"dashboard_view": True},
                "description": "Protocol status request"
            },
            {
                "query": "Should I provide liquidity to the USDC/ETH pool right now?",
                "context": {"user_id": "trader_123", "risk_tolerance": "medium"},
                "description": "Trading advice request"
            },
            {
                "query": "Compare the performance of stable vs volatile pools this week",
                "context": {"analysis_type": "comparative"},
                "description": "Market analysis query"
            }
        ]
        
        print(f"\n📊 Processing {len(example_queries)} example queries...\n")
        
        for i, example in enumerate(example_queries, 1):
            print(f"Query {i}: {example['description']}")
            print(f"🔍 \"{example['query']}\"")
            
            try:
                # Process the query
                response = await brain.process_query(
                    query=example["query"],
                    context=example["context"],
                    user_id=example["context"].get("user_id", f"demo_user_{i}")
                )
                
                # Display results
                print(f"📝 Response (confidence: {response.confidence:.2f}):")
                print(f"   {response.response[:200]}...")
                if response.sources:
                    print(f"📚 Sources: {len(response.sources)} knowledge items")
                if response.suggestions:
                    print(f"💡 Suggestions: {', '.join(response.suggestions[:2])}")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
            
            print()  # Blank line for readability
        
        # Show system health
        print("🔍 System Health Check:")
        health = await brain.get_system_health()
        print(f"   Status: {health['system_status']}")
        print(f"   Uptime: {health['uptime']}")
        print(f"   Queries processed: {health['metrics']['total_queries']}")
        print(f"   Success rate: {health['metrics']['success_rate']:.1%}")
        print(f"   Average response time: {health['metrics']['avg_response_time']:.2f}s")


async def demonstrate_conversation_flow():
    """
    Demonstrate a conversation flow with context maintenance.
    """
    print("\n💬 Demonstrating Conversation Flow...")
    
    config = BrainConfig(
        memory_config={"api_key": "demo_key"},
        protocol_config={"quicknode_url": "demo_url"},
        intelligence_config={"gemini_api_key": "demo_key"}
    )
    
    async with AerodromeBrain(config).lifespan() as brain:
        
        user_id = "conversation_demo_user"
        
        conversation_queries = [
            "What is Aerodrome?",
            "How do I start using it?",
            "What are the risks?",
            "Can you recommend some pools for a beginner?",
            "How much should I invest initially?"
        ]
        
        print("🗣️ Simulating conversation with context...")
        
        for i, query in enumerate(conversation_queries, 1):
            print(f"\nUser: {query}")
            
            try:
                response = await brain.process_query(
                    query=query,
                    context={"conversation_turn": i, "user_level": "beginner"},
                    user_id=user_id
                )
                
                print(f"Brain: {response.response[:150]}...")
                print(f"[Confidence: {response.confidence:.2f}]")
                
            except Exception as e:
                print(f"Brain: Sorry, I encountered an error: {e}")


async def demonstrate_monitoring():
    """
    Demonstrate system monitoring and health tracking.
    """
    print("\n📊 Demonstrating System Monitoring...")
    
    config = BrainConfig(
        memory_config={"api_key": "demo_key"},
        protocol_config={"quicknode_url": "demo_url"},
        intelligence_config={"gemini_api_key": "demo_key"},
        health_check_interval=5,  # Fast health checks for demo
        metrics_update_interval=10
    )
    
    async with AerodromeBrain(config).lifespan() as brain:
        
        print("⏱️ Monitoring system for 30 seconds...")
        
        start_time = datetime.now()
        
        # Monitor for 30 seconds
        while (datetime.now() - start_time).total_seconds() < 30:
            
            # Process a query every 5 seconds
            if int((datetime.now() - start_time).total_seconds()) % 5 == 0:
                test_query = "What's the protocol status?"
                
                try:
                    await brain.process_query(
                        query=test_query,
                        user_id="monitoring_demo"
                    )
                    print("✅ Query processed successfully")
                    
                except Exception as e:
                    print(f"❌ Query failed: {e}")
            
            # Check health every 10 seconds
            if int((datetime.now() - start_time).total_seconds()) % 10 == 0:
                health = await brain.get_system_health()
                print(f"🔍 System Status: {health['system_status']}")
                
                # Show component health
                healthy_components = sum(
                    1 for comp in health['component_health'].values()
                    if comp['status'] in ['ready', 'busy']
                )
                total_components = len(health['component_health'])
                print(f"🧩 Components: {healthy_components}/{total_components} healthy")
            
            await asyncio.sleep(1)  # Check every second
        
        print("✅ Monitoring demo completed")


async def demonstrate_error_handling():
    """
    Demonstrate error handling and recovery.
    """
    print("\n⚠️ Demonstrating Error Handling...")
    
    config = BrainConfig(
        memory_config={"api_key": "demo_key"},
        protocol_config={"quicknode_url": "demo_url"},
        intelligence_config={"gemini_api_key": "demo_key"}
    )
    
    async with AerodromeBrain(config).lifespan() as brain:
        
        # Test various error scenarios
        error_scenarios = [
            {
                "query": "",  # Empty query
                "description": "Empty query"
            },
            {
                "query": "x" * 10000,  # Very long query
                "description": "Extremely long query"
            },
            {
                "query": "What is the price of pool 0xinvalidaddress?",
                "description": "Invalid address"
            },
            {
                "query": "Should I buy 1000000 ETH right now?",
                "description": "Unrealistic trading query"
            }
        ]
        
        for scenario in error_scenarios:
            print(f"Testing: {scenario['description']}")
            
            try:
                response = await brain.process_query(
                    query=scenario["query"],
                    user_id="error_test_user"
                )
                
                print(f"✅ Handled gracefully: {response.response[:100]}...")
                print(f"   Confidence: {response.confidence:.2f}")
                
            except Exception as e:
                print(f"❌ Unhandled error: {e}")
            
            print()


async def main():
    """
    Main demonstration function.
    """
    print("🚀 Aerodrome Brain Usage Examples")
    print("=" * 50)
    
    try:
        # Demonstrate different aspects of the brain system
        await demonstrate_query_types()
        await demonstrate_conversation_flow() 
        await demonstrate_monitoring()
        await demonstrate_error_handling()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Set up your environment variables (API keys)")
        print("2. Configure the brain for your specific needs")
        print("3. Integrate with your application or interface")
        print("4. Monitor system performance and adjust as needed")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demonstrations interrupted by user")
    except Exception as e:
        print(f"\n💥 Demonstration error: {e}")
        print("This is expected in demo mode without real API keys")


if __name__ == "__main__":
    # Run the demonstrations
    asyncio.run(main())