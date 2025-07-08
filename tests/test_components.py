#!/usr/bin/env python3
"""
Test script to validate the modular RAG chatbot components
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_configuration():
    """Test configuration loading and validation."""
    print("🔧 Testing configuration...")
    
    try:
        from src.config import Config
        
        # Test basic configuration
        print(f"✓ OpenAI Model: {Config.OPENAI_MODEL}")
        print(f"✓ Embedding Model: {Config.EMBEDDING_MODEL}")
        print(f"✓ Data Path: {Config.DATA_PATH}")
        print(f"✓ Persist Directory: {Config.CHROMA_PERSIST_DIR}")
        
        # Test metadata validation
        test_metadata = {
            "string": "test",
            "number": 42,
            "boolean": True,
            "none_value": None,
            "list": ["item1", "item2"],
            "dict": {"key": "value"}
        }
        
        sanitized = Config.validate_metadata(test_metadata)
        print(f"✓ Metadata validation: {sanitized}")
        
        print("✅ Configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_processor():
    """Test data processing functionality."""
    print("\n📊 Testing data processor...")
    
    try:
        from src.data.processor import DataProcessor
        
        processor = DataProcessor()
        
        # Test data loading
        data = processor.load_and_process_data()
        print(f"✓ Loaded {len(data)} FAQ entries")
        
        # Test document preparation
        documents = processor.prepare_documents_for_embedding(data)
        print(f"✓ Prepared {len(documents)} documents for embedding")
        
        # Test metadata format
        sample_doc = documents[0]
        metadata = sample_doc["metadata"]
        print(f"✓ Sample metadata: {metadata}")
        
        # Verify metadata types
        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(f"Invalid metadata type for {key}: {type(value)}")
        
        print("✅ Data processor tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False

def test_database_manager():
    """Test database manager functionality."""
    print("\n🗄️ Testing database manager...")
    
    try:
        from src.rag.database import DatabaseManager
        
        # Test with default configuration (should work)
        db_manager = DatabaseManager()
        
        # Test client creation
        client = db_manager.client
        print("✓ Database client created")
        
        # Test embedding function
        ef = db_manager.embedding_function
        print("✓ Embedding function created")
        
        # Test stats
        stats = db_manager.get_collection_stats()
        print(f"✓ Database stats: {stats}")
        
        print("✅ Database manager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database manager test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running RAG Chatbot Component Tests\n")
    
    tests = [
        test_configuration,
        test_data_processor,
        test_database_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n📈 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modular RAG chatbot is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
