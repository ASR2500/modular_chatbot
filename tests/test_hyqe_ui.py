#!/usr/bin/env python3
"""Test script to verify HyQE UI functionality works correctly."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag.database import DatabaseManager
from src.rag.engine import RAGEngine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hyqe_ui_functionality():
    """Test that HyQE results will display properly in the UI."""
    
    # Initialize components
    db_manager = DatabaseManager()
    rag_engine = RAGEngine(db_manager)
    
    # Test query that should return HyQE results
    test_query = "How do I convert a string to integer in Python?"
    
    logger.info(f"Testing query: '{test_query}'")
    
    # Query both collections (this is what the UI does)
    try:
        results = db_manager.query_both_collections(test_query, n_results=5)
        
        logger.info(f"Retrieved {len(results)} results")
        
        hyqe_results = [r for r in results if r.get('metadata', {}).get('generation_method') == 'hyqe_batch']
        regular_results = [r for r in results if r.get('metadata', {}).get('generation_method') != 'hyqe_batch']
        
        logger.info(f"HyQE results: {len(hyqe_results)}")
        logger.info(f"Regular results: {len(regular_results)}")
        
        # Check HyQE results structure
        if hyqe_results:
            logger.info("\\n=== HyQE Results Analysis ===")
            
            for i, result in enumerate(hyqe_results[:3]):  # Check first 3
                metadata = result.get('metadata', {})
                
                # Check what UI expects
                source_question = metadata.get("source_question") or metadata.get("question", "N/A")
                chunk_content = metadata.get("chunk_content", metadata.get("answer", "N/A"))
                
                logger.info(f"\\nHyQE Result {i+1}:")
                logger.info(f"  Generated Question: {source_question}")
                logger.info(f"  Source Chunk: {chunk_content[:100]}..." if len(chunk_content) > 100 else f"  Source Chunk: {chunk_content}")
                logger.info(f"  Metadata keys: {list(metadata.keys())}")
                
                # Check if UI will show "N/A"
                if source_question == "N/A":
                    logger.error(f"  ❌ UI will show 'N/A' for question!")
                else:
                    logger.info(f"  ✅ UI will show the question properly")
                
                if chunk_content == "N/A":
                    logger.error(f"  ❌ UI will show 'N/A' for source chunk!")
                else:
                    logger.info(f"  ✅ UI will show the source chunk properly")
        
        else:
            logger.warning("No HyQE results found for the test query")
            
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hyqe_ui_functionality()
