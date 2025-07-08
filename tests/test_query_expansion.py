"""Test script for query expansion functionality."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.query_expansion.processor import QueryExpansionProcessor
from src.query_expansion_config import QueryExpansionConfig
from dotenv import load_dotenv

load_dotenv()

def test_query_expansion():
    """Test the query expansion functionality."""
    print("🔍 Testing Query Expansion Module")
    print("=" * 50)
    
    # Test configuration
    try:
        config = QueryExpansionConfig()
        config.validate()
        print("✅ Configuration validated successfully")
        
        # Display config
        print("\n📋 Configuration:")
        display_config = config.get_display_config()
        for key, value in display_config.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Test processor initialization
    try:
        processor = QueryExpansionProcessor(config)
        print("\n✅ Query expansion processor initialized successfully")
        
        # Get processor stats
        stats = processor.get_stats()
        print(f"📊 Processor stats: {stats['model']} (temp: {stats['temperature']})")
        
    except Exception as e:
        print(f"❌ Processor initialization failed: {e}")
        return False
    
    # Test query expansion
    test_queries = [
        "How do I create a list in Python?",
        "What is a lambda function?",
        "How to handle exceptions in Python?",
        "Python dictionary methods",
        "Installing packages with pip"
    ]
    
    print("\n🧪 Testing Query Expansion:")
    print("-" * 30)
    
    for query in test_queries:
        try:
            print(f"\nOriginal: {query}")
            expanded = processor.expand_query(query, num_expansions=3)
            
            if expanded:
                print(f"✅ Generated {len(expanded)} expansions:")
                for i, exp in enumerate(expanded, 1):
                    print(f"  {i}. {exp}")
            else:
                print("⚠️  No expansions generated")
            
        except Exception as e:
            print(f"❌ Error expanding query '{query}': {e}")
    
    # Test cache functionality
    print("\n💾 Testing Cache:")
    print("-" * 15)
    
    try:
        # Test same query twice
        query = "How do I create a list in Python?"
        
        # First call
        result1 = processor.expand_query(query, num_expansions=2)
        print(f"First call: {len(result1)} expansions")
        
        # Second call (should use cache)
        result2 = processor.expand_query(query, num_expansions=2)
        print(f"Second call: {len(result2)} expansions")
        
        # Check cache stats
        cache_stats = processor.get_cache_stats()
        print(f"Cache stats: {cache_stats}")
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
    
    # Test Python-specific expansion
    print("\n🐍 Testing Python-specific expansion:")
    print("-" * 35)
    
    try:
        python_query = "How to use pandas dataframes?"
        python_expanded = processor.expand_with_python_context(python_query, num_expansions=3)
        
        if python_expanded:
            print(f"✅ Python-focused expansions for '{python_query}':")
            for i, exp in enumerate(python_expanded, 1):
                print(f"  {i}. {exp}")
        else:
            print("⚠️  No Python-specific expansions generated")
            
    except Exception as e:
        print(f"❌ Python-specific expansion failed: {e}")
    
    # Test expansion suggestions
    print("\n💡 Testing Expansion Suggestions:")
    print("-" * 30)
    
    try:
        suggestions = processor.get_expansion_suggestions("Python functions")
        
        if suggestions:
            print("✅ Strategy-based suggestions:")
            for strategy, expansions in suggestions.items():
                print(f"  {strategy}: {expansions}")
        else:
            print("⚠️  No suggestions generated")
            
    except Exception as e:
        print(f"❌ Expansion suggestions failed: {e}")
    
    print("\n🎉 Query Expansion Test Complete!")
    return True

if __name__ == "__main__":
    success = test_query_expansion()
    exit(0 if success else 1)
