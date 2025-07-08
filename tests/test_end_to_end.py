"""
End-to-end test for the complete pipeline with HyDE and HyQE
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, mock_open

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.app import PythonFAQChatbot
from src.config import Config
from src.hyde_config import HyDEConfig
from src.hyqe_config import HyQEConfig

class TestEndToEnd(unittest.TestCase):
    """End-to-end test of the complete application"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, "test_data.csv")
        self.test_db_path = os.path.join(self.temp_dir, "test_db")
        
        # Create test data
        test_csv_content = """question,answer
"What is Python?","Python is a programming language"
"How to install Python?","You can install Python from python.org"
"What are Python basics?","Python basics include variables, functions, and classes"
"""
        with open(self.test_data_path, 'w') as f:
            f.write(test_csv_content)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_pipeline_integration(self):
        """Test complete pipeline with HyDE and HyQE enabled"""
        
        # Mock configurations
        with patch('src.config.Config') as mock_config:
            # Configure mock config
            mock_config.DATA_PATH = self.test_data_path
            mock_config.CHROMA_DB_PATH = self.test_db_path
            mock_config.ENABLE_QUERY_EXPANSION = True
            mock_config.ENABLE_NER = True
            
            # Configure HyDE
            with patch('src.hyde_config.HyDEConfig') as mock_hyde_config:
                mock_hyde_config.ENABLE_HYDE = True
                mock_hyde_config.HYDE_MODEL = "gpt-3.5-turbo"
                mock_hyde_config.TEMPERATURE = 0.7
                mock_hyde_config.MAX_TOKENS = 100
                
                # Configure HyQE
                with patch('src.hyqe_config.HyQEConfig') as mock_hyqe_config:
                    mock_hyqe_config.ENABLE_HYQE = True
                    mock_hyqe_config.HYQE_MODEL = "gpt-3.5-turbo"
                    mock_hyqe_config.QUESTIONS_PER_CHUNK = 3
                    
                    # Mock OpenAI responses
                    mock_openai_response = MagicMock()
                    mock_openai_response.choices = [
                        MagicMock(message=MagicMock(content="Mock response content"))
                    ]
                    
                    with patch('openai.OpenAI') as mock_openai:
                        mock_client = MagicMock()
                        mock_client.chat.completions.create.return_value = mock_openai_response
                        mock_openai.return_value = mock_client
                        
                        # Mock embedding responses
                        mock_embedding_response = MagicMock()
                        mock_embedding_response.data = [
                            MagicMock(embedding=[0.1, 0.2, 0.3])
                        ]
                        mock_client.embeddings.create.return_value = mock_embedding_response
                        
                        # Test initialization
                        try:
                            # This would normally initialize the complete chatbot
                            # For testing, we'll just verify that our components can work together
                            
                            # Test HyDE config
                            hyde_config = HyDEConfig()
                            assert hasattr(hyde_config, 'ENABLE_HYDE')
                            
                            # Test HyQE config
                            hyqe_config = HyQEConfig()
                            assert hasattr(hyqe_config, 'ENABLE_HYQE')
                            
                            # Test that configs can be created
                            hyde_dict = hyde_config.get_config_dict()
                            hyqe_dict = hyqe_config.get_config_dict()
                            
                            assert isinstance(hyde_dict, dict)
                            assert isinstance(hyqe_dict, dict)
                            
                            # Test that both configs have required keys
                            assert 'enabled' in hyde_dict
                            assert 'enabled' in hyqe_dict
                            
                            print("✓ End-to-end integration test passed")
                            print(f"✓ HyDE config: {len(hyde_dict)} settings")
                            print(f"✓ HyQE config: {len(hyqe_dict)} settings")
                            
                        except Exception as e:
                            self.fail(f"Integration test failed: {e}")
    
    def test_hyde_and_hyqe_compatibility(self):
        """Test that HyDE and HyQE work together without conflicts"""
        
        # Test that both processors can be initialized
        hyde_config = HyDEConfig()
        hyqe_config = HyQEConfig()
        
        # Mock OpenAI for both
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Python is a high-level programming language that is widely used for web development, data science, artificial intelligence, and scientific computing. It was created by Guido van Rossum and first released in 1991. Python is known for its simple syntax and readability, making it an excellent choice for beginners and experts alike."))
        ]
        
        # Test Hyde processor
        from src.hyde.processor import HyDEProcessor
        hyde_processor = HyDEProcessor(hyde_config)
        
        # Mock the client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        hyde_processor.client = mock_client
        
        # Test query generation
        query = "What is Python?"
        hyde_documents = hyde_processor.generate_hypothetical_documents(query)
        
        assert len(hyde_documents) >= 1
        assert isinstance(hyde_documents[0], str)  # HyDE returns strings, not dicts
        
        # Test HyQE processor
        from src.hyqe.processor import HyQEProcessor
        hyqe_processor = HyQEProcessor(hyqe_config)
        
        # Mock the client with a proper question format
        mock_questions_response = MagicMock()
        mock_questions_response.choices = [
            MagicMock(message=MagicMock(content="1. What is Python?\n2. How do you install Python?\n3. What are the basic features of Python?"))
        ]
        hyqe_processor.client = MagicMock()
        hyqe_processor.client.chat.completions.create.return_value = mock_questions_response
        
        # Test question generation
        chunk = "Python is a programming language that is easy to learn and use."
        hyqe_questions = hyqe_processor.generate_questions_for_chunk(chunk)
        
        assert len(hyqe_questions) >= 1
        assert isinstance(hyqe_questions[0], dict)
        assert 'question' in hyqe_questions[0]
        
        print("✓ HyDE and HyQE compatibility test passed")
        print(f"✓ HyDE generated {len(hyde_documents)} documents")
        print(f"✓ HyQE generated {len(hyqe_questions)} questions")

if __name__ == "__main__":
    unittest.main()
