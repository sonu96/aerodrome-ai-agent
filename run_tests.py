#!/usr/bin/env python3
"""
Test runner script for Aerodrome AI Agent.

This script provides convenient ways to run different test suites locally.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"🧪 {description}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install -r requirements.txt")
        return False


def run_unit_tests():
    """Run unit tests."""
    cmd = [
        "pytest", "tests/",
        "--verbose",
        "-m", "unit or not integration",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-fail-under=70"
    ]
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = [
        "pytest", "tests/",
        "--verbose", 
        "-m", "integration",
        "--cov=src",
        "--cov-append",
        "--cov-report=term-missing"
    ]
    return run_command(cmd, "Integration Tests")


def run_slow_tests():
    """Run slow/performance tests."""
    cmd = [
        "pytest", "tests/",
        "--verbose",
        "-m", "slow",
        "--timeout=600"
    ]
    return run_command(cmd, "Slow/Performance Tests")


def run_all_tests():
    """Run all tests."""
    cmd = [
        "pytest", "tests/",
        "--verbose",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--cov-fail-under=70"
    ]
    return run_command(cmd, "All Tests")


def run_specific_test(test_path):
    """Run a specific test file or function."""
    cmd = [
        "pytest",
        test_path,
        "--verbose",
        "-s",  # Don't capture output
        "--tb=long"  # Detailed tracebacks
    ]
    return run_command(cmd, f"Specific Test: {test_path}")


def run_linting():
    """Run code quality checks."""
    success = True
    
    # Black formatting check
    cmd = ["black", "--check", "--diff", "src/", "tests/"]
    if not run_command(cmd, "Black Formatting Check"):
        success = False
    
    # isort import sorting check
    cmd = ["isort", "--check-only", "--diff", "src/", "tests/"]
    if not run_command(cmd, "Import Sorting Check"):
        success = False
    
    # mypy type checking
    cmd = ["mypy", "src/"]
    if not run_command(cmd, "Type Checking"):
        print("Note: Type checking errors are not blocking")
    
    return success


def run_security_scan():
    """Run security scans."""
    success = True
    
    # Bandit security scan
    cmd = ["bandit", "-r", "src/", "-f", "txt"]
    if not run_command(cmd, "Security Scan (Bandit)"):
        success = False
    
    # Safety vulnerability check
    cmd = ["safety", "check"]
    if not run_command(cmd, "Vulnerability Check (Safety)"):
        success = False
    
    return success


def run_benchmarks():
    """Run performance benchmarks."""
    cmd = [
        "pytest", "tests/",
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-warmup=on"
    ]
    return run_command(cmd, "Performance Benchmarks")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for Aerodrome AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run only unit tests
  python run_tests.py --integration             # Run only integration tests
  python run_tests.py --all                     # Run all tests
  python run_tests.py --lint                    # Run linting checks
  python run_tests.py --security                # Run security scans
  python run_tests.py --benchmarks              # Run performance benchmarks
  python run_tests.py --test tests/test_brain.py # Run specific test file
  python run_tests.py --full                    # Run everything
        """
    )
    
    parser.add_argument(
        "--unit", action="store_true",
        help="Run unit tests"
    )
    parser.add_argument(
        "--integration", action="store_true", 
        help="Run integration tests"
    )
    parser.add_argument(
        "--slow", action="store_true",
        help="Run slow/performance tests"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--lint", action="store_true",
        help="Run linting and code quality checks"
    )
    parser.add_argument(
        "--security", action="store_true",
        help="Run security scans"
    )
    parser.add_argument(
        "--benchmarks", action="store_true",
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--test", type=str,
        help="Run specific test file or function"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run complete test suite (tests + linting + security)"
    )
    parser.add_argument(
        "--coverage-html", action="store_true",
        help="Open coverage report in browser after tests"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Error: tests/ directory not found")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    success = True
    
    print("🚀 Aerodrome AI Agent Test Runner")
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    
    # Run tests based on arguments
    if args.unit:
        success &= run_unit_tests()
    elif args.integration:
        success &= run_integration_tests()
    elif args.slow:
        success &= run_slow_tests()
    elif args.all:
        success &= run_all_tests()
    elif args.test:
        success &= run_specific_test(args.test)
    elif args.lint:
        success &= run_linting()
    elif args.security:
        success &= run_security_scan()
    elif args.benchmarks:
        success &= run_benchmarks()
    elif args.full:
        # Run everything
        success &= run_all_tests()
        success &= run_linting()
        success &= run_security_scan()
    else:
        # Default: run unit tests
        print("No specific test type specified, running unit tests...")
        success &= run_unit_tests()
    
    # Open coverage report if requested
    if args.coverage_html and Path("htmlcov/index.html").exists():
        print("\n📊 Opening coverage report in browser...")
        try:
            import webbrowser
            webbrowser.open(Path("htmlcov/index.html").absolute().as_uri())
        except Exception as e:
            print(f"Could not open browser: {e}")
            print("Coverage report available at: htmlcov/index.html")
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        print("\nTips:")
        print("- Check the error messages above")
        print("- Run specific tests with --test path/to/test.py")
        print("- Use --verbose for more detailed output")
        print("- Check htmlcov/index.html for coverage details")
        sys.exit(1)


if __name__ == "__main__":
    main()