"""Test for initialization issues in the Streamlit app."""

import streamlit as st
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.app import PythonFAQChatbot

def test_initialization_persistence():
    """Test that initialization state persists correctly."""
    print("🧪 Testing Initialization State Persistence")
    
    # Mock session state
    mock_session_state = {}
    
    with patch('streamlit.session_state', mock_session_state):
        # Create chatbot instance
        chatbot = PythonFAQChatbot()
        
        # Check initial state
        assert not chatbot.initialized, "Should not be initialized initially"
        assert chatbot.data_processor is None, "Data processor should be None initially"
        assert chatbot.db_manager is None, "DB manager should be None initially"
        assert chatbot.rag_engine is None, "RAG engine should be None initially"
        
        # Mock the components to avoid actual initialization
        with patch('src.app.DataProcessor') as mock_data_proc, \
             patch('src.app.DatabaseManager') as mock_db_mgr, \
             patch('src.app.RAGEngine') as mock_rag_engine, \
             patch('src.config.Config.validate'):
            
            # Mock the setup_data method
            with patch.object(chatbot, 'setup_data'):
                # Initialize
                chatbot.initialize()
                
                # Check that components are stored in session state
                assert 'data_processor' in mock_session_state
                assert 'db_manager' in mock_session_state
                assert 'rag_engine' in mock_session_state
                assert 'initialization_complete' in mock_session_state
                assert mock_session_state['initialization_complete'] is True
                
                # Check that properties access session state
                assert chatbot.initialized is True
                assert chatbot.data_processor is not None
                assert chatbot.db_manager is not None
                assert chatbot.rag_engine is not None
                
                print("✓ Initialization state stored correctly in session state")
                
                # Test that second initialization doesn't reinitialize
                mock_data_proc.reset_mock()
                mock_db_mgr.reset_mock()
                mock_rag_engine.reset_mock()
                
                chatbot.initialize()
                
                # Should not create new instances
                mock_data_proc.assert_not_called()
                mock_db_mgr.assert_not_called()
                mock_rag_engine.assert_not_called()
                
                print("✓ Second initialization skipped correctly")

def test_session_state_consistency():
    """Test that session state access is consistent."""
    print("\n🧪 Testing Session State Consistency")
    
    mock_session_state = {
        'data_processor': MagicMock(),
        'db_manager': MagicMock(),
        'rag_engine': MagicMock(),
        'initialization_complete': True
    }
    
    with patch('streamlit.session_state', mock_session_state):
        chatbot = PythonFAQChatbot()
        
        # Test property access
        assert chatbot.initialized is True
        assert chatbot.data_processor is mock_session_state['data_processor']
        assert chatbot.db_manager is mock_session_state['db_manager']
        assert chatbot.rag_engine is mock_session_state['rag_engine']
        
        print("✓ Property access matches session state")

def test_initialization_flow():
    """Test the complete initialization flow in the UI method."""
    print("\n🧪 Testing UI Initialization Flow")
    
    mock_session_state = {}
    
    with patch('streamlit.session_state', mock_session_state):
        chatbot = PythonFAQChatbot()
        
        # Mock Streamlit UI components
        with patch('streamlit.spinner'), \
             patch('src.app.DataProcessor') as mock_data_proc, \
             patch('src.app.DatabaseManager') as mock_db_mgr, \
             patch('src.app.RAGEngine') as mock_rag_engine, \
             patch('src.config.Config.validate'):
            
            # Mock the data processing
            mock_data_instance = MagicMock()
            mock_data_proc.return_value = mock_data_instance
            mock_data_instance.load_and_process_data.return_value = [{"question": "test", "answer": "test"}]
            mock_data_instance.prepare_documents_for_embedding.return_value = [{"id": "1", "text": "test"}]
            
            # Mock the database
            mock_db_instance = MagicMock()
            mock_db_mgr.return_value = mock_db_instance
            mock_collection = MagicMock()
            mock_collection.count.return_value = 1  # Already populated
            mock_db_instance.get_or_create_collection.return_value = mock_collection
            
            # Mock the RAG engine
            mock_rag_instance = MagicMock()
            mock_rag_engine.return_value = mock_rag_instance
            
            # Mock UI methods
            chatbot.ui.show_success = MagicMock()
            chatbot.ui.show_info = MagicMock()
            chatbot.ui.show_error = MagicMock()
            
            # Test initialization
            chatbot.initialize_with_ui()
            
            # Check that components were stored in session state
            assert 'data_processor' in mock_session_state
            assert 'db_manager' in mock_session_state
            assert 'rag_engine' in mock_session_state
            assert mock_session_state['initialization_complete'] is True
            
            print("✓ UI initialization flow completed successfully")

if __name__ == "__main__":
    print("🚀 Running Initialization Tests")
    
    try:
        test_initialization_persistence()
        test_session_state_consistency()
        test_initialization_flow()
        
        print("\n✅ All initialization tests passed!")
        print("🎉 The initialization issue has been fixed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
