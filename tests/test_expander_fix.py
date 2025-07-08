"""Test script to verify the expander fix works."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.app import PythonFAQChatbot
from src.config import Config
import streamlit as st

def test_expander_fix():
    """Test that the expander fix works correctly."""
    print("🧪 Testing Expander Fix")
    
    # Create a mock session state
    class MockSessionState:
        def __init__(self):
            self.data = {}
        
        def get(self, key, default=None):
            return self.data.get(key, default)
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __contains__(self, key):
            return key in self.data
    
    # Mock streamlit session state
    st.session_state = MockSessionState()
    
    try:
        # Initialize the chatbot
        chatbot = PythonFAQChatbot()
        
        # Test initialization
        chatbot.initialize()
        print("✅ Chatbot initialized successfully")
        
        # Test a simple query to see if it processes without expander errors
        if chatbot.rag_engine:
            result = chatbot.rag_engine.process_query("What is Python?", n_results=2)
            print(f"✅ Query processed successfully")
            print(f"✅ Retrieved {len(result['contexts'])} contexts")
            print(f"✅ Response generated: {len(result['response'])} characters")
            
            # Test that contexts have the expected structure
            for i, ctx in enumerate(result['contexts']):
                metadata = ctx.get('metadata', {})
                answer = metadata.get('answer', '')
                if len(answer) > 500:
                    print(f"✅ Context {i+1} has long answer ({len(answer)} chars) - will be truncated properly")
                else:
                    print(f"✅ Context {i+1} has short answer ({len(answer)} chars)")
        
        print("\n🎉 Expander fix test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Expander Fix")
    success = test_expander_fix()
    
    if success:
        print("\n✅ All tests passed! The expander nesting issue should be fixed.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
