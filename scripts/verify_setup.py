#!/usr/bin/env python3
"""
Setup verification script

Verifies the project structure is correctly set up without requiring
external dependencies to be installed.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def verify_file_structure():
    """Verify essential files exist"""
    
    essential_files = [
        "pyproject.toml",
        "requirements.txt", 
        ".env.example",
        ".gitignore",
        "README.md",
        "setup.py",
        "src/aerodrome_ai_agent/__init__.py",
        "src/aerodrome_ai_agent/brain/__init__.py",
        "src/aerodrome_ai_agent/brain/config.py",
        "src/aerodrome_ai_agent/brain/state.py",
        "src/aerodrome_ai_agent/brain/core.py",
        "src/aerodrome_ai_agent/memory/__init__.py",
        "src/aerodrome_ai_agent/memory/config.py",
        "src/aerodrome_ai_agent/memory/system.py",
        "src/aerodrome_ai_agent/cdp/__init__.py",
        "src/aerodrome_ai_agent/cdp/config.py",
        "src/aerodrome_ai_agent/cdp/manager.py",
        "src/aerodrome_ai_agent/config/__init__.py",
        "src/aerodrome_ai_agent/config/base.py",
        "src/aerodrome_ai_agent/config/settings.py",
        "src/aerodrome_ai_agent/contracts/__init__.py",
        "src/aerodrome_ai_agent/contracts/addresses.py",
        "src/aerodrome_ai_agent/cli.py",
        "tests/__init__.py",
        "tests/test_basic.py"
    ]
    
    print("🔍 Verifying file structure...")
    missing_files = []
    
    for file_path in essential_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print(f"\n✅ All {len(essential_files)} essential files present")
        return True


def verify_basic_imports():
    """Verify basic configuration classes can be imported"""
    
    print("\n🔍 Verifying basic imports...")
    
    try:
        # Test configuration imports (these don't require external deps)
        from aerodrome_ai_agent.brain.config import BrainConfig
        from aerodrome_ai_agent.memory.config import MemoryConfig  
        from aerodrome_ai_agent.cdp.config import CDPConfig
        from aerodrome_ai_agent.brain.state import create_initial_state
        from aerodrome_ai_agent.contracts.addresses import BASE_TOKENS
        
        print("   ✅ Configuration classes import successfully")
        
        # Test basic instantiation
        brain_config = BrainConfig()
        memory_config = MemoryConfig()
        cdp_config = CDPConfig()
        initial_state = create_initial_state()
        
        print("   ✅ Configuration classes instantiate successfully")
        
        # Test basic validation
        brain_config.validate()
        memory_config.validate()
        cdp_config.validate()
        
        print("   ✅ Configuration validation works")
        
        # Test data structures
        assert len(BASE_TOKENS) > 0
        assert "USDC" in BASE_TOKENS
        assert "WETH" in BASE_TOKENS
        
        print("   ✅ Contract data structures loaded")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def verify_dependencies():
    """Check if requirements.txt has necessary dependencies"""
    
    print("\n🔍 Verifying dependencies list...")
    
    required_deps = [
        "cdp-sdk",
        "langgraph", 
        "mem0ai",
        "openai",
        "pandas",
        "numpy",
        "pydantic",
        "httpx",
        "fastapi",
        "click"
    ]
    
    requirements_file = project_root / "requirements.txt"
    if not requirements_file.exists():
        print("   ❌ requirements.txt not found")
        return False
    
    requirements_text = requirements_file.read_text()
    
    missing_deps = []
    for dep in required_deps:
        if dep not in requirements_text:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"   ❌ Missing dependencies: {missing_deps}")
        return False
    else:
        print(f"   ✅ All {len(required_deps)} required dependencies listed")
        return True


def main():
    """Run all verification checks"""
    
    print("🚀 Aerodrome AI Agent - Setup Verification")
    print("=" * 50)
    
    checks = [
        ("File Structure", verify_file_structure),
        ("Basic Imports", verify_basic_imports), 
        ("Dependencies", verify_dependencies)
    ]
    
    passed = 0
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                print(f"\n❌ {check_name} check failed")
        except Exception as e:
            print(f"\n❌ {check_name} check failed with error: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Results: {passed}/{len(checks)} checks passed")
    
    if passed == len(checks):
        print("✅ Project setup verification successful!")
        print("\n🎉 Next steps:")
        print("   1. Copy .env.example to .env and fill in your API keys")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("   4. Run the agent: python -m aerodrome_ai_agent.cli start")
        return 0
    else:
        print("❌ Project setup verification failed!")
        print("   Please check the errors above and fix them")
        return 1


if __name__ == "__main__":
    sys.exit(main())