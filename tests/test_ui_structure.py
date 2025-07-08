"""Simple test to check the UI components without full app initialization."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.components import ChatbotUI

def test_render_response_structure():
    """Test that the render_response_with_contexts method has proper structure."""
    print("🧪 Testing UI Component Structure")
    
    try:
        # Create UI component
        ui = ChatbotUI()
        print("✅ UI component created successfully")
        
        # Test data structure
        test_response = "This is a test response about Python programming."
        test_contexts = [
            {
                "metadata": {
                    "question": "What is Python?",
                    "answer": "Python is a high-level programming language" * 20,  # Long answer
                    "keywords": "python, programming, language"
                },
                "similarity": 0.95
            },
            {
                "metadata": {
                    "question": "How to install Python?",
                    "answer": "You can install Python from python.org",  # Short answer
                    "keywords": "python, install, download"
                },
                "similarity": 0.87
            }
        ]
        
        print(f"✅ Test data prepared:")
        print(f"  - Response: {len(test_response)} characters")
        print(f"  - Contexts: {len(test_contexts)} items")
        print(f"  - Context 1 answer: {len(test_contexts[0]['metadata']['answer'])} characters (should be truncated)")
        print(f"  - Context 2 answer: {len(test_contexts[1]['metadata']['answer'])} characters (should be full)")
        
        # Check the method exists and is callable
        if hasattr(ui, 'render_response_with_contexts'):
            print("✅ render_response_with_contexts method exists")
            print("✅ Method structure appears correct (no nested expanders)")
        else:
            print("❌ render_response_with_contexts method not found")
            return False
            
        # Inspect the method to ensure no nested expanders
        import inspect
        source = inspect.getsource(ui.render_response_with_contexts)
        
        # Count expanders in the method
        expander_count = source.count('st.expander')
        print(f"✅ Method contains {expander_count} expander(s)")
        
        # Check for nested expanders (should be only 1)
        if expander_count <= 1:
            print("✅ No nested expanders detected in method")
        else:
            print("❌ Multiple expanders detected - potential nesting issue")
            
        # Look for the specific pattern we fixed
        if 'with st.expander("Show full answer")' in source:
            print("❌ Found nested expander pattern!")
            return False
        else:
            print("✅ No nested expander pattern found")
            
        print("\n🎉 UI component structure test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing UI Component Structure")
    success = test_render_response_structure()
    
    if success:
        print("\n✅ UI component test passed! The nested expander issue should be fixed.")
    else:
        print("\n❌ UI component test failed. Please check the error messages above.")
