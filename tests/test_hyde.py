"""Test script for HyDE (Hypothetical Document Embeddings) functionality."""

import os
import sys
import logging
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.hyde.processor import HyDEProcessor
from src.hyde_config import HyDEConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hyde_basic_functionality():
    """Test basic HyDE functionality."""
    print("🧪 Testing HyDE Basic Functionality")
    print("=" * 50)
    
    try:
        # Initialize HyDE processor
        hyde_processor = HyDEProcessor()
        
        # Test queries
        test_queries = [
            "How do I create a list in Python?",
            "What is the difference between append and extend?",
            "How to handle exceptions in Python?",
            "What are Python decorators?",
            "How to read a file in Python?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {query}")
            print("-" * 40)
            
            # Generate single hypothetical document
            hyde_doc = hyde_processor.generate_hypothetical_document(query, domain="python")
            
            if hyde_doc:
                print(f"✅ Generated document ({len(hyde_doc)} characters):")
                print(f"'{hyde_doc[:100]}...' " if len(hyde_doc) > 100 else f"'{hyde_doc}'")
            else:
                print("❌ Failed to generate document")
            
            # Test batch mode if enabled
            if hyde_processor.config.BATCH_MODE:
                print(f"\n🔄 Batch mode test:")
                hyde_docs = hyde_processor.generate_hypothetical_documents(query, domain="python")
                print(f"Generated {len(hyde_docs)} documents")
                
        # Test cache statistics
        cache_stats = hyde_processor.get_cache_stats()
        if cache_stats:
            print(f"\n📊 Cache Statistics:")
            print(f"  - Total requests: {cache_stats['total_requests']}")
            print(f"  - Cache hits: {cache_stats['hits']}")
            print(f"  - Cache misses: {cache_stats['misses']}")
            print(f"  - Hit rate: {cache_stats['hit_rate_percent']}%")
            print(f"  - Cache size: {cache_stats['cache_size']}")
        
        print("\n✅ HyDE basic functionality test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing HyDE: {str(e)}")
        logger.error(f"HyDE test failed: {str(e)}")
        return False
    
    return True

def test_hyde_config_options():
    """Test different HyDE configuration options."""
    print("\n🧪 Testing HyDE Configuration Options")
    print("=" * 50)
    
    try:
        # Test different configurations
        configs = [
            {"temperature": 0.2, "max_tokens": 100, "answer_style": "concise"},
            {"temperature": 0.5, "max_tokens": 200, "answer_style": "detailed"},
            {"temperature": 0.3, "max_tokens": 150, "answer_style": "steps"},
            {"temperature": 0.2, "max_tokens": 120, "answer_style": "bullets"},
        ]
        
        query = "How do I iterate through a dictionary in Python?"
        
        for i, config in enumerate(configs, 1):
            print(f"\n📋 Configuration {i}: {config}")
            print("-" * 40)
            
            # Create config with custom settings
            hyde_config = HyDEConfig()
            hyde_config.TEMPERATURE = config["temperature"]
            hyde_config.MAX_TOKENS = config["max_tokens"]
            hyde_config.ANSWER_STYLE = config["answer_style"]
            
            # Test with this configuration
            hyde_processor = HyDEProcessor(hyde_config)
            hyde_doc = hyde_processor.generate_hypothetical_document(query, domain="python")
            
            if hyde_doc:
                print(f"✅ Generated document ({len(hyde_doc)} characters)")
                print(f"Preview: {hyde_doc[:80]}...")
            else:
                print("❌ Failed to generate document")
        
        print("\n✅ HyDE configuration options test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing HyDE configurations: {str(e)}")
        logger.error(f"HyDE configuration test failed: {str(e)}")
        return False
    
    return True

def test_hyde_batch_mode():
    """Test HyDE batch mode functionality."""
    print("\n🧪 Testing HyDE Batch Mode")
    print("=" * 50)
    
    try:
        # Create config with batch mode enabled
        hyde_config = HyDEConfig()
        hyde_config.BATCH_MODE = True
        hyde_config.BATCH_SIZE = 3
        
        hyde_processor = HyDEProcessor(hyde_config)
        
        query = "What are Python list comprehensions?"
        print(f"📝 Query: {query}")
        print("-" * 40)
        
        # Generate multiple hypothetical documents
        hyde_docs = hyde_processor.generate_hypothetical_documents(query, domain="python")
        
        print(f"✅ Generated {len(hyde_docs)} documents in batch mode:")
        for i, doc in enumerate(hyde_docs, 1):
            print(f"\n📄 Document {i} ({len(doc)} characters):")
            print(f"  {doc[:100]}...")
        
        print("\n✅ HyDE batch mode test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing HyDE batch mode: {str(e)}")
        logger.error(f"HyDE batch mode test failed: {str(e)}")
        return False
    
    return True

def main():
    """Run all HyDE tests."""
    print("🚀 Starting HyDE Tests")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key to run these tests")
        return
    
    tests = [
        test_hyde_basic_functionality,
        test_hyde_config_options,
        test_hyde_batch_mode
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED with exception: {str(e)}")
    
    print(f"\n📊 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! HyDE implementation is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
