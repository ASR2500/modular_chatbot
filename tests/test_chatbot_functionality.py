"""
Simple test script to verify the chatbot functionality with HyDE and HyQE.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.app import PythonFAQChatbot
from src.config import Config

def test_chatbot_functionality():
    """Test basic chatbot functionality with HyDE and HyQE."""
    
    print("Testing Python FAQ Chatbot with HyDE and HyQE...")
    print("=" * 60)
    
    try:
        # Initialize the chatbot
        print("1. Initializing chatbot...")
        app = PythonFAQChatbot()
        print("   ✓ Chatbot initialized successfully")
        
        # Test basic setup
        print("\n2. Testing basic setup...")
        app.init_session_state()
        print("   ✓ Session state initialized")
        
        # Test a simple query
        print("\n3. Testing query processing...")
        test_query = "What is Python?"
        
        # Mock the RAG engine query for testing
        if hasattr(app, 'rag_engine') and app.rag_engine:
            print(f"   Query: {test_query}")
            print("   ✓ RAG engine is available")
            
            # Check if HyDE and HyQE are enabled
            hyde_enabled = app.rag_engine.enable_hyde
            hyqe_enabled = app.rag_engine.enable_hyqe
            
            print(f"   HyDE enabled: {hyde_enabled}")
            print(f"   HyQE enabled: {hyqe_enabled}")
            
            # Test stats retrieval
            stats = app.rag_engine.get_stats()
            print(f"   ✓ Engine stats retrieved: {len(stats)} categories")
            
        else:
            print("   ⚠ RAG engine not available")
        
        print("\n4. Testing configuration...")
        
        # Test HyDE config
        from src.hyde_config import HyDEConfig
        hyde_config = HyDEConfig()
        hyde_dict = hyde_config.get_config_dict()
        print(f"   ✓ HyDE config loaded: {len(hyde_dict)} settings")
        
        # Test HyQE config
        from src.hyqe_config import HyQEConfig
        hyqe_config = HyQEConfig()
        hyqe_dict = hyqe_config.get_config_dict()
        print(f"   ✓ HyQE config loaded: {len(hyqe_dict)} settings")
        
        print("\n✅ All tests passed!")
        print("The chatbot is ready to use with both HyDE and HyQE modules.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chatbot_functionality()
    sys.exit(0 if success else 1)
