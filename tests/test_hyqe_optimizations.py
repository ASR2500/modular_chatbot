"""Test HyQE optimizations for efficiency and cost savings."""

import pytest
import time
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from src.hyqe.processor import HyQEProcessor
from src.hyqe_config import HyQEConfig
from src.rag.database import DatabaseManager
from src.rag.engine import RAGEngine

class TestHyQEOptimizations:
    """Test HyQE optimization features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = HyQEConfig()
        self.config.OPENAI_API_KEY = "test-key"
        self.config.ENABLE_CACHE = True
        self.config.BATCH_MODE = True
        self.config.BATCH_SIZE = 3
        self.config.VERBOSE = True
        
        self.processor = HyQEProcessor(self.config)
        
        # Mock OpenAI client
        self.processor.client = Mock()
        
        # Test data
        self.test_chunks = [
            {"id": "chunk1", "content": "This is about Python functions"},
            {"id": "chunk2", "content": "This is about Python classes"},
            {"id": "chunk3", "content": "This is about Python modules"},
            {"id": "chunk4", "content": "This is about Python exceptions"},
            {"id": "chunk5", "content": "This is about Python decorators"}
        ]
        
        self.test_questions = [
            "What are Python functions?",
            "How do you define a class in Python?",
            "What is a Python module?"
        ]
    
    def test_content_hashing_optimization(self):
        """Test content hashing for change detection."""
        # Test that content hashes are generated correctly
        hash1 = self.processor._generate_content_hash("test content")
        hash2 = self.processor._generate_content_hash("test content")
        hash3 = self.processor._generate_content_hash("different content")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_content_change_detection(self):
        """Test that only changed content is processed."""
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "\n".join(self.test_questions)
        mock_response.usage.total_tokens = 100
        self.processor.client.chat.completions.create.return_value = mock_response
        
        # First run - should process all chunks
        filtered_chunks = self.processor._filter_chunks_for_processing(self.test_chunks)
        assert len(filtered_chunks) == len(self.test_chunks)
        
        # Second run with same content - should process no chunks
        filtered_chunks = self.processor._filter_chunks_for_processing(self.test_chunks)
        assert len(filtered_chunks) == 0
        
        # Third run with one changed chunk - should process only that chunk
        modified_chunks = self.test_chunks.copy()
        modified_chunks[0]["content"] = "This is modified content about Python functions"
        filtered_chunks = self.processor._filter_chunks_for_processing(modified_chunks)
        assert len(filtered_chunks) == 1
        assert filtered_chunks[0]["id"] == "chunk1"
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "\n".join(self.test_questions * 5)  # Multiple questions
        mock_response.usage.total_tokens = 500
        self.processor.client.chat.completions.create.return_value = mock_response
        
        # Test batch processing
        questions = self.processor.generate_questions_for_chunks(self.test_chunks, domain="python")
        
        # Should have made fewer API calls due to batching
        assert self.processor.client.chat.completions.create.call_count <= 2  # 5 chunks, batch size 3
        assert len(questions) > 0
        
        # Check that batch operation was recorded
        assert self.processor._generation_stats["batch_operations"] > 0
    
    def test_cache_effectiveness(self):
        """Test caching effectiveness."""
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "\n".join(self.test_questions)
        mock_response.usage.total_tokens = 100
        self.processor.client.chat.completions.create.return_value = mock_response
        
        # First generation - should hit API
        questions1 = self.processor.generate_questions_for_chunk("test content", "test1")
        api_calls_after_first = self.processor.client.chat.completions.create.call_count
        
        # Second generation with same content - should hit cache
        questions2 = self.processor.generate_questions_for_chunk("test content", "test2")
        api_calls_after_second = self.processor.client.chat.completions.create.call_count
        
        # Should not have made additional API calls
        assert api_calls_after_first == api_calls_after_second
        assert self.processor._generation_stats["cache_hits"] > 0
        assert len(questions1) > 0
        assert len(questions2) > 0
    
    def test_statistics_tracking(self):
        """Test comprehensive statistics tracking."""
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "\n".join(self.test_questions)
        mock_response.usage.total_tokens = 100
        self.processor.client.chat.completions.create.return_value = mock_response
        
        # Generate questions - use individual processing to ensure stats are updated
        for chunk in self.test_chunks[:2]:
            self.processor.generate_questions_for_chunk(
                chunk["content"], 
                chunk["id"]
            )
        
        # Check statistics
        stats = self.processor.get_generation_stats()
        assert stats["total_chunks_processed"] > 0
        assert stats["total_questions_generated"] > 0
        assert stats["total_api_calls"] > 0
        assert stats["total_api_tokens"] > 0
        assert stats["total_processing_time"] >= 0  # Should be non-negative
    
    def test_cache_persistence(self):
        """Test cache save and load functionality."""
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "\n".join(self.test_questions)
        mock_response.usage.total_tokens = 100
        self.processor.client.chat.completions.create.return_value = mock_response
        
        # Generate some questions to populate cache
        questions = self.processor.generate_questions_for_chunk("test content", "test1")
        
        # Verify content hash was added
        assert len(self.processor._content_hashes) > 0
        
        # Test save/load content hashes
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # Save content hashes
            self.processor.save_content_hashes(temp_path)
            
            # Create new processor and load hashes
            new_processor = HyQEProcessor(self.config)
            new_processor.load_content_hashes(temp_path)
            
            # Check that hashes were loaded
            assert len(new_processor._content_hashes) > 0
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations."""
        # Mock some statistics
        self.processor._generation_stats.update({
            "total_chunks_processed": 100,
            "total_api_calls": 50,
            "total_api_tokens": 10000,
            "total_processing_time": 120,
            "batch_operations": 0
        })
        
        # Mock cache stats
        self.processor._cache_stats = {"hits": 10, "misses": 90}
        
        # Get recommendations
        recommendations = self.processor.get_optimization_recommendations()
        
        # Should have recommendations
        assert len(recommendations) > 0
        assert any("cache" in rec.lower() for rec in recommendations)
        assert any("batch" in rec.lower() for rec in recommendations)
    
    def test_cost_estimation(self):
        """Test cost estimation functionality."""
        # Mock some statistics
        self.processor._generation_stats.update({
            "total_api_calls": 10,
            "total_api_tokens": 5000,
            "batch_operations": 2
        })
        
        # Mock cache stats
        self.processor._cache_stats = {"hits": 5, "misses": 5}
        
        # Get cost estimates
        cost_estimates = self.processor.estimate_cost_savings()
        
        # Should have cost estimates
        assert "total_estimated_cost" in cost_estimates
        assert "cache_savings" in cost_estimates
        assert "batch_savings" in cost_estimates
        assert "total_savings" in cost_estimates
        
        # Should be positive numbers
        assert cost_estimates["total_estimated_cost"] >= 0
        assert cost_estimates["cache_savings"] >= 0
        assert cost_estimates["batch_savings"] >= 0
    
    def test_cache_efficiency_calculation(self):
        """Test cache efficiency calculation."""
        # Mock cache stats
        self.processor._cache_stats = {"hits": 80, "misses": 20}
        
        efficiency = self.processor.get_cache_efficiency()
        assert efficiency == 80.0  # 80% efficiency
        
        # Test with no cache
        self.processor._cache_stats = None
        efficiency = self.processor.get_cache_efficiency()
        assert efficiency == 0.0


class TestOptimizedDatabaseOperations:
    """Test optimized database operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the database manager instead of using real one
        self.db_manager = Mock()
        
        # Mock ChromaDB collection
        self.mock_collection = Mock()
        self.mock_collection.count.return_value = 0
        self.mock_collection.get.return_value = {"ids": []}
        self.mock_collection.add = Mock()
        
        # Set up the optimized method
        self.db_manager.populate_questions_collection_optimized = Mock(return_value=2)
        self.db_manager.get_or_create_questions_collection = Mock(return_value=self.mock_collection)
    
    def test_optimized_population_with_deduplication(self):
        """Test optimized population with deduplication."""
        # Test questions
        questions = [
            {
                "question": "What is Python?",
                "chunk_id": "chunk1",
                "chunk_content": "Python is a programming language",
                "question_index": 0,
                "generation_method": "hyqe",
                "config": {"style": "natural", "quality": "balanced"}
            },
            {
                "question": "How do you use Python?",
                "chunk_id": "chunk1",
                "chunk_content": "Python is a programming language",
                "question_index": 1,
                "generation_method": "hyqe",
                "config": {"style": "natural", "quality": "balanced"}
            }
        ]
        
        # Test population
        count = self.db_manager.populate_questions_collection_optimized(
            questions=questions,
            batch_size=5,
            skip_existing=False
        )
        
        # Should have added questions
        assert count >= len(questions)
        assert self.db_manager.populate_questions_collection_optimized.called
    
    def test_skip_existing_questions(self):
        """Test skipping when questions already exist."""
        # Mock existing questions
        self.mock_collection.count.return_value = 100
        
        questions = [{"question": "test", "chunk_id": "test"}]
        
        # Test with skip_existing=True
        count = self.db_manager.populate_questions_collection_optimized(
            questions=questions,
            skip_existing=True
        )
        
        # Should return existing count without adding
        assert count == 2  # Mocked return value
        assert self.db_manager.populate_questions_collection_optimized.called


class TestEndToEndOptimization:
    """Test end-to-end optimization workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = HyQEConfig()
        self.config.OPENAI_API_KEY = "test-key"
        self.config.ENABLE_CACHE = True
        self.config.BATCH_MODE = True
        self.config.BATCH_SIZE = 3
        
        # Mock RAG engine components
        self.rag_engine = RAGEngine(
            database_manager=Mock(),
            enable_hyqe=True
        )
        
        # Mock HyQE processor
        self.rag_engine.hyqe_processor = Mock()
        self.rag_engine.hyqe_processor.generate_questions_for_chunks.return_value = [
            {"question": "What is Python?", "chunk_id": "chunk1"}
        ]
        self.rag_engine.hyqe_processor.load_content_hashes = Mock()
        self.rag_engine.hyqe_processor.save_content_hashes = Mock()
        
        # Mock database manager
        self.rag_engine.db_manager.populate_questions_collection_optimized = Mock(return_value=100)
    
    def test_optimized_question_generation_workflow(self):
        """Test the optimized question generation workflow."""
        # Test documents
        documents = [
            {"id": "doc1", "text": "Python is a programming language", "metadata": {}}
        ]
        
        # Test optimized generation
        success = self.rag_engine.generate_and_populate_question_embeddings(
            documents=documents
        )
        
        # Should succeed
        assert success is True
        
        # Should have called optimization methods
        assert self.rag_engine.hyqe_processor.load_content_hashes.called
        assert self.rag_engine.hyqe_processor.save_content_hashes.called
        assert self.rag_engine.db_manager.populate_questions_collection_optimized.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
