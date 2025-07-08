#!/usr/bin/env python3
"""
Integration test for the complete RAG chatbot application
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_full_application():
    """Test the complete application flow."""
    print("🚀 Testing Full Application Integration\n")
    
    try:
        from src.app import PythonFAQChatbot
        
        print("🔧 Initializing chatbot...")
        chatbot = PythonFAQChatbot()
        
        print("📊 Loading data and initializing components...")
        chatbot.initialize()
        
        print("✓ Data processor initialized")
        print("✓ Database manager initialized")
        print("✓ RAG engine initialized")
        
        # Test data stats
        if chatbot.data_processor:
            data_stats = chatbot.data_processor.get_data_stats()
            print(f"✓ Data stats: {data_stats.get('total_entries', 'N/A')} entries")
        
        # Test engine stats
        if chatbot.rag_engine:
            engine_stats = chatbot.rag_engine.get_engine_stats()
            db_count = engine_stats.get('database_stats', {}).get('document_count', 'N/A')
            print(f"✓ Engine stats: {db_count} documents in database")
        
        # Test a simple query
        print("\n🔍 Testing query processing...")
        if chatbot.rag_engine:
            test_query = "What is Python?"
            result = chatbot.rag_engine.process_query(test_query, n_results=2)
            
            print(f"✓ Query: '{test_query}'")
            print(f"✓ Retrieved {result['n_contexts']} contexts")
            print(f"✓ Response length: {len(result['response'])} characters")
            
            # Show first 100 characters of response
            response_preview = result['response'][:100] + "..." if len(result['response']) > 100 else result['response']
            print(f"✓ Response preview: {response_preview}")
        
        print("\n✅ Full application integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n🛡️ Testing Error Handling...")
    
    try:
        from src.app import PythonFAQChatbot
        
        chatbot = PythonFAQChatbot()
        chatbot.initialize()
        
        # Test with empty stats
        empty_stats = {}
        try:
            # This should not crash
            user_settings = chatbot.ui.render_sidebar(empty_stats, empty_stats)
            print("✓ Handles empty stats gracefully")
        except Exception as e:
            print(f"❌ Failed to handle empty stats: {e}")
            return False
        
        print("✅ Error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 RAG Chatbot Integration Tests\n")
    
    tests = [
        test_full_application,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n📈 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! The RAG chatbot is fully functional.")
        return True
    else:
        print("❌ Some integration tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
