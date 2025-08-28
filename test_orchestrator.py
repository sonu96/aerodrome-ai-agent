#!/usr/bin/env python3
"""
Simple test script for the Aerodrome AI Agent Orchestrator

This script provides a basic test to verify that all components can be initialized
and work together correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.aerodrome_ai_agent.main import AerodromeAgent, validate_environment, test_agent


async def test_orchestrator():
    """Test the orchestrator system"""
    
    print("🧪 Testing Aerodrome AI Agent Orchestrator")
    print("=" * 50)
    
    # Test 1: Environment validation
    print("\n1. Validating environment...")
    validation = validate_environment()
    
    if validation['valid']:
        print("✅ Environment validation passed")
        print(f"   Environment: {validation['environment']}")
        print(f"   Network: {validation['network']}")
    else:
        print(f"❌ Environment validation failed: {validation['error']}")
        print("   Please check your .env file")
        return False
    
    # Test 2: Agent initialization
    print("\n2. Testing agent initialization...")
    try:
        agent = AerodromeAgent()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False
    
    # Test 3: Configuration validation
    print("\n3. Testing configuration...")
    try:
        config = agent.validate_configuration()
        if config['valid']:
            print("✅ Configuration validated")
        else:
            print(f"❌ Configuration invalid: {config['error']}")
            return False
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # Test 4: Component test run
    print("\n4. Testing components with short run...")
    try:
        success = await test_agent(cycles=1)
        if success:
            print("✅ Component test completed successfully")
        else:
            print("❌ Component test failed")
            return False
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Orchestrator is ready to use.")
    return True


def main():
    """Main test function"""
    try:
        success = asyncio.run(test_orchestrator())
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()