# Aerodrome AI Agent Testing Framework

This directory contains a comprehensive testing framework for the Aerodrome AI Agent, covering unit tests, integration tests, mocks, fixtures, and CI/CD automation.

## 🏗️ Architecture

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── pytest.ini              # Pytest settings and markers
├── .coveragerc             # Coverage configuration
├── run_tests.py            # Local test runner script
├── README.md               # This file
│
├── fixtures/               # Test data and scenarios
│   ├── __init__.py
│   ├── market_data.py      # Market scenarios and price data
│   ├── memory_data.py      # Memory system test data
│   └── contract_data.py    # Blockchain/contract test data
│
├── mocks/                  # Mock objects for external services
│   ├── __init__.py
│   ├── cdp_mocks.py        # CDP SDK mocks
│   ├── memory_mocks.py     # Memory system mocks
│   └── llm_mocks.py        # OpenAI/LLM mocks
│
├── unit/                   # Unit test directories (if needed)
├── integration/            # Integration test directories (if needed)
│
├── test_brain.py          # Brain module tests
├── test_memory.py         # Memory system tests
├── test_cdp.py           # CDP SDK integration tests
└── test_orchestrator.py  # End-to-end integration tests
```

## 🧪 Test Categories

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Marker**: `@pytest.mark.unit`
- **Coverage**: Core functionality, configuration, state management
- **Execution**: Fast, no external dependencies

### Integration Tests  
- **Purpose**: Test component interactions and workflows
- **Marker**: `@pytest.mark.integration`
- **Coverage**: Cross-component communication, data flow
- **Execution**: Moderate speed, uses mocks for external services

### Slow Tests
- **Purpose**: Performance, long-running operations, comprehensive scenarios
- **Marker**: `@pytest.mark.slow`
- **Coverage**: End-to-end workflows, performance benchmarks
- **Execution**: Slow, comprehensive testing

### External Tests
- **Purpose**: Tests requiring real external services (optional)
- **Marker**: `@pytest.mark.external` 
- **Coverage**: Real API integration (when credentials available)
- **Execution**: Requires real API keys, network access

## 🔧 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure test environment is set up
export TESTING=true
export LOG_LEVEL=DEBUG
```

### Running Tests

```bash
# Run all tests
python run_tests.py --all

# Run specific test categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --integration   # Integration tests only
python run_tests.py --slow          # Performance tests only

# Run specific test files
python run_tests.py --test tests/test_brain.py
pytest tests/test_memory.py::TestMemoryConfig::test_default_config

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Development Workflow

```bash
# Quick feedback loop during development
pytest tests/test_brain.py -v -s

# Run linting checks
python run_tests.py --lint

# Run security scans
python run_tests.py --security

# Complete test suite
python run_tests.py --full
```

## 🎯 Test Coverage Goals

- **Minimum Coverage**: 70% (CI/CD enforced)
- **Target Coverage**: 80%+
- **Critical Components**: >90% coverage
  - Brain core logic
  - Memory operations
  - CDP transactions
  - Error handling

## 🔍 Test Structure

### Test Classes
Each test file is organized into logical test classes:

```python
class TestComponentConfiguration:
    """Test configuration and initialization."""

class TestCoreComponentFunctionality:
    """Test main component operations."""
    
class TestErrorHandling:
    """Test error scenarios and recovery."""
    
class TestIntegration:
    """Test component interactions."""
```

### Fixtures and Mocks
- **Shared Fixtures**: Defined in `conftest.py`
- **Mock Objects**: Realistic simulations of external services
- **Test Data**: Comprehensive scenarios in `fixtures/`

### Async Testing
- All async functions properly tested with `@pytest.mark.asyncio`
- Event loop management handled automatically
- Timeout protection for long-running tests

## 📊 CI/CD Pipeline

### GitHub Actions Workflow
`.github/workflows/test.yml` provides:

- **Multi-Python Testing**: Python 3.9, 3.10, 3.11
- **Service Dependencies**: Qdrant vector database
- **Parallel Execution**: Matrix strategy for efficiency
- **Comprehensive Reporting**: Coverage, security, performance
- **Artifact Storage**: Test results, coverage reports

### Workflow Stages
1. **Setup**: Environment, dependencies, services
2. **Linting**: Black, isort, mypy
3. **Unit Tests**: Fast component tests
4. **Integration Tests**: Cross-component tests  
5. **Security Scan**: Bandit, Safety checks
6. **Performance**: Benchmarks (main branch only)
7. **Build Verification**: Package building
8. **Reporting**: Results summary and artifacts

### Branch Protection
- **Main Branch**: All tests must pass
- **Pull Requests**: Test results posted as comments
- **Coverage Enforcement**: Minimum 70% required

## 🎭 Mocking Strategy

### External Services
All external services are mocked by default:

```python
# CDP SDK mocking
mock_cdp_manager = MockCDPManager()
mock_cdp_manager.set_should_fail(True, "NetworkError")

# Memory system mocking  
mock_memory = MockMemorySystem()
mock_memory.add_memory(test_memory_data)

# OpenAI API mocking
mock_openai_client.set_custom_response("analysis", "Test AI response")
```

### Mock Configuration
- **Deterministic Behavior**: Consistent results across test runs
- **Error Simulation**: Configurable failure scenarios
- **State Management**: Proper mock state isolation
- **Performance**: Fast execution without real API calls

## 🛡️ Security Testing

### Static Analysis
- **Bandit**: Security linting for Python code
- **Safety**: Vulnerability scanning for dependencies
- **Results**: Uploaded as CI/CD artifacts

### Best Practices
- No hardcoded secrets in test code
- Environment variable validation
- Input sanitization testing
- Error message security review

## 📈 Performance Testing

### Benchmarking
```python
@pytest.mark.benchmark
def test_memory_recall_performance(benchmark):
    result = benchmark(memory_system.recall_memories, context)
    assert len(result) > 0
```

### Performance Targets
- **Memory Recall**: < 100ms for 1000 memories
- **Brain Cycles**: < 30s end-to-end
- **CDP Operations**: < 5s simulation + execution
- **Pattern Extraction**: < 1s for 100 experiences

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### Async Test Failures
```python
# Ensure proper async/await usage
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

#### Mock State Pollution
```python
# Reset mocks between tests
@pytest.fixture(autouse=True)
def reset_mocks():
    mock_object.reset()
    yield
    mock_object.clear()
```

#### Coverage Issues
```bash
# Run with coverage debugging
pytest --cov=src --cov-report=term-missing --cov-debug=trace
```

### Debug Mode
```bash
# Verbose output with no capture
pytest tests/ -v -s --tb=long

# Debug specific test
pytest tests/test_brain.py::TestBrain::test_specific -v -s --pdb
```

## 🔄 Continuous Improvement

### Test Metrics
- **Execution Time**: Monitor and optimize slow tests
- **Flaky Tests**: Identify and fix unstable tests  
- **Coverage Trends**: Track coverage improvements
- **Error Patterns**: Analyze common failure modes

### Adding New Tests
1. **Choose Category**: Unit, integration, or slow
2. **Add Markers**: Use appropriate pytest markers
3. **Mock Dependencies**: Use existing mocks or create new ones
4. **Follow Patterns**: Maintain consistent test structure
5. **Update Documentation**: Keep this README current

### Best Practices
- **Test Names**: Descriptive and specific
- **Test Isolation**: No dependencies between tests
- **Data Management**: Use fixtures for test data
- **Error Testing**: Include failure scenarios
- **Documentation**: Comment complex test logic

## 📚 References

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [GitHub Actions](https://docs.github.com/en/actions)

## 🤝 Contributing

When adding tests:

1. Follow the existing structure and patterns
2. Add appropriate markers and fixtures
3. Include both success and failure scenarios
4. Maintain or improve test coverage
5. Update documentation as needed
6. Ensure tests pass in CI/CD pipeline

For questions or suggestions, please open an issue or discussion in the repository.