"""
Test HyQE (Hypothetical Question Embeddings) functionality
"""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hyqe.processor import HyQEProcessor
from hyqe_config import HyQEConfig


def test_hyqe_basic_functionality():
    """Test basic HyQE functionality"""
    # Test config
    config = HyQEConfig()
    assert config.ENABLE_HYQE is True
    assert config.QUESTIONS_PER_CHUNK == 3
    assert config.QUESTION_STYLE == "natural"
    
    # Test processor
    processor = HyQEProcessor(config)
    
    # Mock the OpenAI client
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="1. What is Python?\n2. How to install Python?\n3. What are Python basics?"))
    ]
    
    # Mock the client at the processor level
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    processor.client = mock_client
    
    # Test question generation
    chunk = "Python is a programming language that is easy to learn and use."
    questions = processor.generate_questions_for_chunk(chunk)
    
    assert len(questions) >= 1  # Should generate at least 1 question
    # Check that questions are returned as dictionaries with required keys
    for question in questions:
        assert isinstance(question, dict)
        assert 'question' in question
        assert 'chunk_content' in question
        assert 'chunk_id' in question
        assert 'generation_method' in question
        assert question['generation_method'] == 'hyqe'
    
    # Verify OpenAI was called with correct parameters
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args['model'] == config.HYQE_MODEL
    assert call_args['temperature'] == config.TEMPERATURE
    assert call_args['max_tokens'] == config.MAX_TOKENS
        
    return True


def test_hyqe_config_options():
    """Test HyQE configuration options"""
    config = HyQEConfig()
    
    # Test all configuration options
    assert hasattr(config, 'ENABLE_HYQE')
    assert hasattr(config, 'HYQE_MODEL')
    assert hasattr(config, 'QUESTIONS_PER_CHUNK')
    assert hasattr(config, 'QUESTION_STYLE')
    assert hasattr(config, 'QUALITY')
    assert hasattr(config, 'TEMPERATURE')
    assert hasattr(config, 'MAX_TOKENS')
    assert hasattr(config, 'INCLUDE_CHUNK_SUMMARY')
    assert hasattr(config, 'BATCH_MODE')
    assert hasattr(config, 'ENABLE_CACHE')
    
    # Test configuration types
    assert isinstance(config.ENABLE_HYQE, bool)
    assert isinstance(config.HYQE_MODEL, str)
    assert isinstance(config.QUESTIONS_PER_CHUNK, int)
    assert isinstance(config.QUESTION_STYLE, str)
    assert isinstance(config.QUALITY, str)
    assert isinstance(config.TEMPERATURE, float)
    assert isinstance(config.MAX_TOKENS, int)
    assert isinstance(config.INCLUDE_CHUNK_SUMMARY, bool)
    assert isinstance(config.BATCH_MODE, bool)
    assert isinstance(config.ENABLE_CACHE, bool)
    
    # Test value ranges
    assert 0.0 <= config.TEMPERATURE <= 1.0
    assert config.MAX_TOKENS > 0
    assert config.QUESTIONS_PER_CHUNK > 0
    
    return True


def test_hyqe_batch_mode():
    """Test HyQE batch processing mode"""
    config = HyQEConfig()
    
    processor = HyQEProcessor(config)
    
    # Mock response for batch processing
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="1. Question 1\n2. Question 2\n3. Question 3"))
    ]
    
    # Mock the client at the processor level
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    processor.client = mock_client
    
    # Test processing multiple chunks
    chunks = [
        "First chunk about Python basics",
        "Second chunk about Python advanced topics"
    ]
    
    results = []
    for chunk in chunks:
        questions = processor.generate_questions_for_chunk(chunk)
        results.append(questions)
    
    assert len(results) == 2
    assert len(results[0]) >= 1  # Should generate at least 1 question per chunk
    assert len(results[1]) >= 1
        
    return True


def test_hyqe_different_styles():
    """Test different HyQE question styles"""
    styles = ["natural", "instructional", "faq", "exam-style"]
    
    for style in styles:
        config = HyQEConfig()
        config.QUESTION_STYLE = style
        
        processor = HyQEProcessor(config)
        
        # Test that different styles are configured
        assert config.QUESTION_STYLE == style
        
        # Test that the style is in the available styles
        assert style in config.QUESTION_STYLES
        
    return True


if __name__ == "__main__":
    test_hyqe_basic_functionality()
    test_hyqe_config_options()
    test_hyqe_batch_mode()
    test_hyqe_different_styles()
    print("All HyQE tests passed!")
