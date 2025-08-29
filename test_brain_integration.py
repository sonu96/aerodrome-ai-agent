#!/usr/bin/env python3
"""
Comprehensive integration test for the Aerodrome Brain system.

This script tests the integration between all brain components:
- Core orchestrator
- Knowledge base
- Query handler
- Confidence scoring
- Memory management
- Protocol integration
- AI intelligence

Run this script to verify that all components work together correctly.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.brain import (
        AerodromeBrain,
        BrainConfig,
        QueryContext,
        QueryType,
        SystemStatus
    )
    print("✓ Successfully imported brain components")
except ImportError as e:
    print(f"✗ Failed to import brain components: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrainIntegrationTester:
    """Comprehensive integration tester for the Aerodrome Brain"""
    
    def __init__(self):
        self.brain: AerodromeBrain = None
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.start_time = datetime.now()
    
    def create_test_config(self) -> BrainConfig:
        """Create a test configuration for the brain"""
        return BrainConfig(
            # Use environment variables if available, otherwise use mock values
            memory_config={
                "api_key": os.getenv("MEM0_API_KEY", "test_mem0_key"),
                "enable_graph": False,  # Disable for testing
                "client_config": {"batch_size": 10}
            },
            protocol_config={
                "quicknode_url": os.getenv("QUICKNODE_URL", "https://test-quicknode-url.com"),
                "client_config": {"timeout": 30}
            },
            intelligence_config={
                "gemini_api_key": os.getenv("GEMINI_API_KEY", "test_gemini_key"),
                "model": "gemini-2.0-flash-001",
                "client_config": {"temperature": 0.7}
            },
            confidence_config={
                "factor_weights": {
                    "data_source_reliability": 0.25,
                    "historical_accuracy": 0.20,
                    "recency": 0.20,
                    "corroboration": 0.15,
                    "sample_size": 0.15,
                    "prediction_success_rate": 0.05
                }
            },
            health_check_interval=10,  # Faster for testing
            metrics_update_interval=20,
            max_concurrent_queries=3,  # Lower for testing
            log_level="INFO",
            enable_detailed_logging=True
        )
    
    async def test_brain_initialization(self) -> bool:
        """Test brain initialization and component setup"""
        test_name = "brain_initialization"
        print(f"\n🧠 Testing {test_name}...")
        
        try:
            config = self.create_test_config()
            self.brain = AerodromeBrain(config)
            
            # Test initialization
            start_time = time.time()
            await self.brain.initialize()
            init_time = time.time() - start_time
            
            # Check system status
            if self.brain.status == SystemStatus.HEALTHY:
                self.test_results[test_name] = {
                    "status": "PASSED",
                    "initialization_time": init_time,
                    "components_initialized": len(self.brain.component_health),
                    "details": "Brain initialized successfully"
                }
                print(f"✓ Brain initialization successful ({init_time:.2f}s)")
                return True
            else:
                self.test_results[test_name] = {
                    "status": "FAILED",
                    "error": f"Brain status: {self.brain.status.value}",
                    "initialization_time": init_time
                }
                print(f"✗ Brain initialization failed: {self.brain.status.value}")
                return False
                
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "exception_type": type(e).__name__
            }
            print(f"✗ Brain initialization error: {e}")
            return False
    
    async def test_component_health(self) -> bool:
        """Test component health monitoring"""
        test_name = "component_health"
        print(f"\n🔍 Testing {test_name}...")
        
        try:
            if not self.brain:
                raise RuntimeError("Brain not initialized")
            
            # Get system health
            health = await self.brain.get_system_health()
            
            healthy_components = 0
            total_components = len(health["component_health"])
            
            for component_name, component_health in health["component_health"].items():
                if component_health["status"] in ["ready", "busy"]:
                    healthy_components += 1
                    print(f"  ✓ {component_name}: {component_health['status']}")
                else:
                    print(f"  ✗ {component_name}: {component_health['status']} - {component_health.get('last_error', 'No error')}")
            
            health_ratio = healthy_components / max(total_components, 1)
            
            self.test_results[test_name] = {
                "status": "PASSED" if health_ratio >= 0.7 else "FAILED",
                "healthy_components": healthy_components,
                "total_components": total_components,
                "health_ratio": health_ratio,
                "system_status": health["system_status"],
                "details": health
            }
            
            if health_ratio >= 0.7:
                print(f"✓ Component health good ({healthy_components}/{total_components} healthy)")
                return True
            else:
                print(f"✗ Component health poor ({healthy_components}/{total_components} healthy)")
                return False
                
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "exception_type": type(e).__name__
            }
            print(f"✗ Component health test error: {e}")
            return False
    
    async def test_basic_queries(self) -> bool:
        """Test basic query processing"""
        test_name = "basic_queries"
        print(f"\n💬 Testing {test_name}...")
        
        test_queries = [
            {
                "query": "What is Aerodrome?",
                "expected_type": QueryType.EDUCATIONAL,
                "description": "Educational query"
            },
            {
                "query": "Show me protocol metrics",
                "expected_type": QueryType.PROTOCOL_STATUS,
                "description": "Protocol status query"
            },
            {
                "query": "What are the top pools?",
                "expected_type": QueryType.POOL_INQUIRY,
                "description": "Pool inquiry"
            },
            {
                "query": "How does voting work?",
                "expected_type": QueryType.VOTING_INQUIRY,
                "description": "Voting question"
            }
        ]
        
        try:
            if not self.brain:
                raise RuntimeError("Brain not initialized")
            
            query_results = []
            total_processing_time = 0
            
            for i, test_query in enumerate(test_queries):
                print(f"  Testing query {i+1}: {test_query['description']}")
                
                query_context = QueryContext(
                    query=test_query["query"],
                    user_id="test_user",
                    context={"test_mode": True}
                )
                
                start_time = time.time()
                response = await self.brain.process_query(
                    query_context.query,
                    context=query_context.context,
                    user_id=query_context.user_id
                )
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                query_result = {
                    "query": test_query["query"],
                    "response_length": len(response.response),
                    "confidence": response.confidence,
                    "processing_time": processing_time,
                    "sources_count": len(response.sources),
                    "has_response": len(response.response) > 0,
                    "metadata": response.metadata
                }
                
                query_results.append(query_result)
                
                if response.response and len(response.response) > 10:
                    print(f"    ✓ Query processed successfully (confidence: {response.confidence:.2f})")
                else:
                    print(f"    ✗ Query processing failed or empty response")
            
            successful_queries = sum(1 for r in query_results if r["has_response"])
            success_rate = successful_queries / len(test_queries)
            avg_processing_time = total_processing_time / len(test_queries)
            avg_confidence = sum(r["confidence"] for r in query_results) / len(query_results)
            
            self.test_results[test_name] = {
                "status": "PASSED" if success_rate >= 0.5 else "FAILED",
                "successful_queries": successful_queries,
                "total_queries": len(test_queries),
                "success_rate": success_rate,
                "avg_processing_time": avg_processing_time,
                "avg_confidence": avg_confidence,
                "query_results": query_results
            }
            
            if success_rate >= 0.5:
                print(f"✓ Basic queries test passed ({successful_queries}/{len(test_queries)} successful)")
                return True
            else:
                print(f"✗ Basic queries test failed ({successful_queries}/{len(test_queries)} successful)")
                return False
                
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "exception_type": type(e).__name__
            }
            print(f"✗ Basic queries test error: {e}")
            return False
    
    async def test_knowledge_base_sync(self) -> bool:
        """Test knowledge base data synchronization"""
        test_name = "knowledge_base_sync"
        print(f"\n📚 Testing {test_name}...")
        
        try:
            if not self.brain or not self.brain.knowledge_base:
                raise RuntimeError("Knowledge base not available")
            
            # Get initial stats
            initial_stats = await self.brain.knowledge_base.get_knowledge_stats()
            initial_items = initial_stats.get("total_items", 0)
            
            print(f"  Initial knowledge items: {initial_items}")
            
            # Trigger sync
            start_time = time.time()
            await self.brain.knowledge_base.sync_protocol_data()
            sync_time = time.time() - start_time
            
            # Get updated stats
            updated_stats = await self.brain.knowledge_base.get_knowledge_stats()
            final_items = updated_stats.get("total_items", 0)
            
            print(f"  Final knowledge items: {final_items}")
            
            items_added = max(0, final_items - initial_items)
            
            self.test_results[test_name] = {
                "status": "PASSED",
                "initial_items": initial_items,
                "final_items": final_items,
                "items_added": items_added,
                "sync_time": sync_time,
                "stats": updated_stats
            }
            
            print(f"✓ Knowledge base sync completed ({items_added} items added)")
            return True
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "exception_type": type(e).__name__
            }
            print(f"✗ Knowledge base sync error: {e}")
            return False
    
    async def test_confidence_scoring(self) -> bool:
        """Test confidence scoring system"""
        test_name = "confidence_scoring"
        print(f"\n🎯 Testing {test_name}...")
        
        try:
            if not self.brain or not self.brain.confidence_scorer:
                raise RuntimeError("Confidence scorer not available")
            
            # Test with a mock memory item
            from src.brain import MemoryItem, MemoryCategory, DataSourceType
            from datetime import datetime
            
            test_item = MemoryItem(
                id="test_confidence_item",
                category=MemoryCategory.POOL_PERFORMANCE,
                data={
                    "pool": "test_pool",
                    "tvl": 1000000,
                    "volume_24h": 500000,
                    "sample_size": 100
                },
                confidence=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source_type=DataSourceType.API_ENDPOINT,
                corroborating_sources={"quicknode", "test_source"}
            )
            
            # Calculate confidence
            start_time = time.time()
            confidence, factors = await self.brain.confidence_scorer.calculate_confidence(
                test_item,
                metadata={
                    "provider": "quicknode",
                    "sample_size": 100,
                    "uptime": 0.99
                }
            )
            calc_time = time.time() - start_time
            
            # Test retention decision
            test_item.confidence = confidence
            should_retain = self.brain.confidence_scorer.should_retain_memory(test_item)
            
            self.test_results[test_name] = {
                "status": "PASSED" if 0 <= confidence <= 1 else "FAILED",
                "confidence_score": confidence,
                "calculation_time": calc_time,
                "should_retain": should_retain,
                "factors": {
                    "data_source_reliability": factors.data_source_reliability,
                    "historical_accuracy": factors.historical_accuracy,
                    "recency": factors.recency,
                    "corroboration": factors.corroboration,
                    "sample_size": factors.sample_size,
                    "prediction_success_rate": factors.prediction_success_rate
                }
            }
            
            if 0 <= confidence <= 1:
                print(f"✓ Confidence scoring working (score: {confidence:.3f})")
                return True
            else:
                print(f"✗ Confidence scoring failed (invalid score: {confidence})")
                return False
                
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "exception_type": type(e).__name__
            }
            print(f"✗ Confidence scoring test error: {e}")
            return False
    
    async def test_performance_metrics(self) -> bool:
        """Test system performance metrics"""
        test_name = "performance_metrics"
        print(f"\n📊 Testing {test_name}...")
        
        try:
            if not self.brain:
                raise RuntimeError("Brain not initialized")
            
            # Get system health (includes metrics)
            health = await self.brain.get_system_health()
            metrics = health.get("metrics", {})
            
            # Check for required metrics
            required_metrics = [
                "total_queries", "successful_queries", "failed_queries", 
                "success_rate", "avg_response_time"
            ]
            
            missing_metrics = [m for m in required_metrics if m not in metrics]
            
            self.test_results[test_name] = {
                "status": "PASSED" if not missing_metrics else "FAILED",
                "metrics_available": len(metrics),
                "required_metrics": len(required_metrics),
                "missing_metrics": missing_metrics,
                "metrics": metrics,
                "uptime": health.get("uptime", "unknown")
            }
            
            if not missing_metrics:
                print(f"✓ Performance metrics complete ({len(metrics)} metrics)")
                return True
            else:
                print(f"✗ Performance metrics incomplete (missing: {missing_metrics})")
                return False
                
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "error": str(e),
                "exception_type": type(e).__name__
            }
            print(f"✗ Performance metrics test error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        print("🚀 Starting Aerodrome Brain Integration Tests")
        print(f"📅 Test started at: {self.start_time}")
        
        tests = [
            self.test_brain_initialization,
            self.test_component_health,
            self.test_basic_queries,
            self.test_knowledge_base_sync,
            self.test_confidence_scoring,
            self.test_performance_metrics,
        ]
        
        passed_tests = 0
        
        for test in tests:
            try:
                result = await test()
                if result:
                    passed_tests += 1
            except Exception as e:
                print(f"✗ Test {test.__name__} crashed: {e}")
        
        # Cleanup
        if self.brain:
            try:
                await self.brain.shutdown()
                print("\n🧹 Brain shutdown completed")
            except Exception as e:
                print(f"⚠️ Brain shutdown error: {e}")
        
        # Summary
        total_tests = len(tests)
        success_rate = passed_tests / total_tests
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        summary = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "duration_seconds": duration,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "test_results": self.test_results
        }
        
        print(f"\n📋 Test Summary:")
        print(f"   Tests passed: {passed_tests}/{total_tests}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Duration: {duration:.1f}s")
        
        if success_rate >= 0.8:
            print("🎉 Integration tests PASSED!")
        elif success_rate >= 0.5:
            print("⚠️ Integration tests PARTIALLY PASSED")
        else:
            print("❌ Integration tests FAILED")
        
        return summary


async def main():
    """Main test runner"""
    tester = BrainIntegrationTester()
    
    try:
        # Run all tests
        results = await tester.run_all_tests()
        
        # Save results to file
        results_file = f"brain_integration_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        success_rate = results["test_summary"]["success_rate"]
        sys.exit(0 if success_rate >= 0.8 else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        if tester.brain:
            await tester.brain.shutdown()
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    # Set up event loop for async execution
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTests interrupted")
        sys.exit(2)