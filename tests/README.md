# Testing Guide

Comprehensive testing guide for the Advanced RAG Chatbot Framework.

## 🧪 Test Overview

The framework includes extensive testing coverage for all modules and integration points:

- **Unit Tests**: Individual module functionality
- **Integration Tests**: Module interaction testing  
- **End-to-End Tests**: Complete pipeline testing
- **Performance Tests**: Speed and resource usage
- **Configuration Tests**: Parameter validation

## 🏃‍♂️ Running Tests

### Run All Tests
```bash
# From project root
cd tests
python -m pytest -v

# With coverage report
python -m pytest --cov=src --cov-report=html -v

# Run specific test file
python -m pytest test_hyde.py -v

# Run with specific markers
python -m pytest -m "unit" -v
python -m pytest -m "integration" -v
```

### Test Categories

#### Unit Tests
- `test_hyde.py` - HyDE module tests
- `test_hyqe.py` - HyQE module tests  
- `test_query_expansion.py` - Query expansion tests
- `test_ner.py` - NER module tests
- `test_components.py` - UI component tests

#### Integration Tests
- `test_integration.py` - Module integration tests
- `test_rag_ner.py` - RAG engine with NER integration
- `test_hyqe_data_flow.py` - HyQE data flow tests

#### End-to-End Tests
- `test_end_to_end.py` - Complete pipeline tests
- `test_chatbot_functionality.py` - Full chatbot tests
- `test_app_functionality.py` - Application-level tests

#### Performance Tests
- `test_hyqe_optimizations.py` - HyQE performance tests
- `test_optimized_method.py` - Method optimization tests

## 📊 Test Coverage

### Current Coverage
- **Overall**: 85%+
- **Core RAG Engine**: 90%+
- **Individual Modules**: 80%+
- **Integration Points**: 75%+

### Coverage Report
```bash
# Generate detailed coverage report
python -m pytest --cov=src --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

## 🔧 Test Configuration

### Test Environment Setup
```bash
# Create test environment file
cp .env.example .env.test

# Set test-specific variables
OPENAI_API_KEY=test_key_or_mock
CHROMA_PERSIST_DIRECTORY=./test_chroma_db
ENABLE_TESTING=true
```

### Pytest Configuration (`pytest.ini`)
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    slow: Slow-running tests
addopts = 
    -v
    --strict-markers
    --tb=short
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

## 📝 Test Examples

### Unit Test Example
```python
# test_hyde.py
import pytest
from unittest.mock import Mock, patch
from src.hyde.processor import HyDEProcessor
from src.hyde_config import HyDEConfig

class TestHyDEProcessor:
    
    @pytest.fixture
    def hyde_processor(self):
        """Create HyDE processor with test config."""
        config = HyDEConfig()
        config.ENABLE_CACHE = False  # Disable cache for testing
        return HyDEProcessor(config)
    
    def test_generate_hypothetical_document(self, hyde_processor):
        """Test single document generation."""
        with patch.object(hyde_processor, '_generate_document') as mock_gen:
            mock_gen.return_value = "Test hypothetical document"
            
            result = hyde_processor.generate_hypothetical_document(
                "How to create a list?", 
                domain="python"
            )
            
            assert result == "Test hypothetical document"
            mock_gen.assert_called_once()
    
    def test_batch_generation(self, hyde_processor):
        """Test batch document generation."""
        hyde_processor.config.BATCH_MODE = True
        hyde_processor.config.BATCH_SIZE = 2
        
        with patch.object(hyde_processor, '_generate_document') as mock_gen:
            mock_gen.return_value = "Test document"
            
            results = hyde_processor.generate_hypothetical_documents("test query")
            
            assert len(results) == 2
            assert all(doc == "Test document" for doc in results)
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        config = HyDEConfig()
        config.ENABLE_CACHE = True
        processor = HyDEProcessor(config)
        
        with patch.object(processor, '_generate_document') as mock_gen:
            mock_gen.return_value = "Cached document"
            
            # First call should generate
            result1 = processor.generate_hypothetical_document("test")
            # Second call should use cache
            result2 = processor.generate_hypothetical_document("test")
            
            assert result1 == result2
            mock_gen.assert_called_once()  # Only called once due to cache
```

### Integration Test Example
```python
# test_integration.py
import pytest
from src.rag.engine import RAGEngine
from src.rag.database import DatabaseManager
from unittest.mock import Mock, patch

class TestRAGEngineIntegration:
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        db_manager = Mock(spec=DatabaseManager)
        db_manager.query_both_collections.return_value = [
            {
                "content": "Python lists are created with square brackets",
                "similarity": 0.9,
                "source": "document_embedding",
                "metadata": {"category": "basics"}
            }
        ]
        return db_manager
    
    def test_query_expansion_integration(self, mock_db_manager):
        """Test query expansion integration with RAG engine."""
        rag_engine = RAGEngine(
            database_manager=mock_db_manager,
            enable_query_expansion=True,
            enable_hyde=False,
            enable_hyqe=False,
            enable_ner=False
        )
        
        # Mock query expansion
        with patch.object(rag_engine.query_expansion_processor, 'expand_query') as mock_expand:
            mock_expand.return_value = ["How to make lists?", "Creating Python lists"]
            
            with patch.object(rag_engine, 'generate_response') as mock_response:
                mock_response.return_value = "Test response"
                
                result = rag_engine.process_query(
                    "How to create lists?",
                    use_query_expansion=True,
                    return_expansion_details=True
                )
                
                assert "expanded_queries" in result
                assert len(result["expanded_queries"]) == 2
                mock_expand.assert_called_once()
    
    def test_multi_module_integration(self, mock_db_manager):
        """Test multiple modules working together."""
        rag_engine = RAGEngine(
            database_manager=mock_db_manager,
            enable_query_expansion=True,
            enable_hyde=True,
            enable_ner=True
        )
        
        with patch.object(rag_engine.query_expansion_processor, 'expand_query') as mock_qe:
            mock_qe.return_value = ["expanded query"]
            
            with patch.object(rag_engine.hyde_processor, 'generate_hypothetical_documents') as mock_hyde:
                mock_hyde.return_value = ["hypothetical doc"]
                
                with patch.object(rag_engine.ner_processor, 'extract_entities') as mock_ner:
                    mock_ner.return_value = []
                    
                    with patch.object(rag_engine, 'generate_response') as mock_response:
                        mock_response.return_value = "Integrated response"
                        
                        result = rag_engine.process_query("test query")
                        
                        assert result["response"] == "Integrated response"
                        mock_qe.assert_called()
                        mock_hyde.assert_called()
```

### End-to-End Test Example
```python
# test_end_to_end.py
import pytest
import tempfile
import os
from src.rag.engine import RAGEngine
from src.rag.database import DatabaseManager
from src.data.processor import DataProcessor

class TestEndToEndWorkflow:
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield os.path.join(temp_dir, "test_chroma_db")
    
    @pytest.fixture
    def test_data(self):
        """Create test dataset."""
        return [
            {
                "question": "How to create a list in Python?",
                "answer": "Use square brackets: my_list = []",
                "category": "basics"
            },
            {
                "question": "What is a dictionary?", 
                "answer": "A dictionary is a collection of key-value pairs",
                "category": "data_structures"
            }
        ]
    
    def test_complete_workflow(self, temp_db_path, test_data):
        """Test complete workflow from data loading to response generation."""
        
        # 1. Initialize components
        db_manager = DatabaseManager(persist_directory=temp_db_path)
        data_processor = DataProcessor()
        
        # 2. Process and load data
        processed_docs = data_processor.process_documents(test_data)
        
        texts = [doc["text"] for doc in processed_docs]
        metadatas = [doc["metadata"] for doc in processed_docs]
        
        db_manager.initialize_collections()
        db_manager.add_documents(texts, metadatas)
        
        # 3. Initialize RAG engine
        rag_engine = RAGEngine(
            database_manager=db_manager,
            enable_query_expansion=True,
            enable_hyde=True
        )
        
        # 4. Test query processing
        result = rag_engine.process_query("How do I make a list?")
        
        # 5. Verify results
        assert result["response"] is not None
        assert len(result["contexts"]) > 0
        assert any("list" in ctx["content"].lower() for ctx in result["contexts"])
        
        # 6. Test with enhancements
        enhanced_result = rag_engine.process_query(
            "Creating lists in Python",
            use_query_expansion=True,
            use_hyde=True,
            return_expansion_details=True
        )
        
        assert "expanded_queries" in enhanced_result
        assert enhanced_result["response"] is not None
```

### Performance Test Example
```python
# test_performance.py
import pytest
import time
import statistics
from src.rag.engine import RAGEngine
from src.rag.database import DatabaseManager

class TestPerformance:
    
    @pytest.fixture
    def rag_engine(self):
        """Create RAG engine for performance testing."""
        db_manager = DatabaseManager()
        return RAGEngine(database_manager=db_manager)
    
    @pytest.mark.performance
    def test_query_processing_speed(self, rag_engine):
        """Test query processing speed."""
        test_queries = [
            "How to create a list?",
            "What is a dictionary?", 
            "How to handle exceptions?",
            "What are Python modules?",
            "How to work with files?"
        ]
        
        response_times = []
        
        for query in test_queries:
            start_time = time.time()
            result = rag_engine.process_query(query)
            end_time = time.time()
            
            response_times.append(end_time - start_time)
            assert result["response"] is not None
        
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        
        # Performance assertions
        assert avg_time < 5.0, f"Average response time too slow: {avg_time:.2f}s"
        assert max_time < 10.0, f"Maximum response time too slow: {max_time:.2f}s"
    
    @pytest.mark.performance
    def test_memory_usage(self, rag_engine):
        """Test memory usage during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple queries
        for i in range(10):
            result = rag_engine.process_query(f"Test query {i}")
            assert result["response"] is not None
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should not increase significantly
        assert memory_increase < 100, f"Memory usage increased too much: {memory_increase:.2f}MB"
```

## 🏷️ Test Markers

Use pytest markers to categorize tests:

```python
# Mark test categories
@pytest.mark.unit
def test_unit_functionality():
    """Unit test."""
    pass

@pytest.mark.integration  
def test_module_integration():
    """Integration test."""
    pass

@pytest.mark.e2e
def test_end_to_end():
    """End-to-end test."""
    pass

@pytest.mark.performance
@pytest.mark.slow
def test_performance_heavy():
    """Performance test that takes time."""
    pass

# Run specific categories
# pytest -m "unit and not slow"
# pytest -m "integration or e2e"
```

## 🔧 Test Utilities

### Mock Helpers
```python
# tests/conftest.py
import pytest
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mocked response"
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

@pytest.fixture
def sample_contexts():
    """Sample contexts for testing."""
    return [
        {
            "content": "Python is a programming language",
            "similarity": 0.9,
            "source": "document_embedding",
            "metadata": {"category": "basics"}
        },
        {
            "content": "Lists are created with square brackets",
            "similarity": 0.8,
            "source": "question_embedding", 
            "metadata": {"category": "data_structures"}
        }
    ]
```

### Test Data Factory
```python
# tests/factories.py
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_document(content="Test content", category="test"):
        """Create a test document."""
        return {
            "question": f"Test question about {category}",
            "answer": content,
            "category": category,
            "metadata": {"source": "test", "difficulty": "easy"}
        }
    
    @staticmethod
    def create_context(content="Test context", similarity=0.8):
        """Create a test context."""
        return {
            "content": content,
            "similarity": similarity,
            "source": "document_embedding",
            "metadata": {"category": "test"}
        }
    
    @staticmethod
    def create_entity(text="TestEntity", label="TEST", confidence=0.9):
        """Create a test entity."""
        from src.ner.processor import Entity
        return Entity(
            text=text,
            label=label, 
            start=0,
            end=len(text),
            confidence=confidence
        )
```

## 🐛 Debugging Tests

### Verbose Output
```bash
# Run with maximum verbosity
pytest -vvv

# Show local variables on failure
pytest --tb=long

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s
```

### Logging in Tests
```python
import logging

def test_with_logging():
    """Test with logging enabled."""
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Your test code here
    # Logs will be visible with pytest -s
```

## 🎯 Test Best Practices

### 1. Test Organization
- Group related tests in classes
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

### 2. Fixtures and Mocking
- Use fixtures for common setup
- Mock external dependencies
- Test both success and failure paths

### 3. Assertions
- Use specific assertions
- Test edge cases
- Verify both positive and negative cases

### 4. Performance
- Mark slow tests appropriately
- Use timeouts for long-running tests
- Monitor resource usage

## 📈 Continuous Integration

### GitHub Actions Example
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

## 📊 Test Reporting

### Coverage Reports
```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term

# Coverage with branch analysis
pytest --cov=src --cov-branch --cov-report=html
```

### Test Results
```bash
# Generate JUnit XML report
pytest --junitxml=test-results.xml

# Generate HTML report
pytest --html=report.html --self-contained-html
```

This testing guide ensures comprehensive coverage and quality assurance for the Advanced RAG Chatbot Framework. Regular testing helps maintain reliability and catch issues early in development.
