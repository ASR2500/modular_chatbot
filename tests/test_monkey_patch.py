#!/usr/bin/env python3
"""Minimal test to add optimized method to DatabaseManager."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Clear any cached modules
for module in list(sys.modules.keys()):
    if module.startswith('src.'):
        del sys.modules[module]

# Import the class
from src.rag.database import DatabaseManager
from typing import List, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)

# Add the method manually to the class
def populate_questions_collection_optimized(self, questions: List[Dict[str, Any]], 
                                          questions_collection_name: str = None,
                                          batch_size: int = 20, 
                                          progress_callback=None,
                                          skip_existing: bool = True):
    """Optimized method to populate questions collection with deduplication."""
    from src.config import Config
    
    collection = self.get_or_create_questions_collection(questions_collection_name)
    
    # Check if collection already has questions and skip if requested
    existing_count = collection.count()
    if skip_existing and existing_count > 0:
        logger.info(f"Questions collection already contains {existing_count} questions, skipping population")
        return existing_count
    
    logger.info(f"Populating questions collection with {len(questions)} questions (existing: {existing_count})")
    
    # Process questions in batches
    added_count = 0
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        
        try:
            # Prepare batch data
            batch_questions = [q["question"] for q in batch]
            batch_ids = [f"q_{q.get('chunk_id', 'unknown')}_{q.get('question_index', i)}" for q in batch]
            batch_metadata = [
                {
                    "question": q["question"],
                    "chunk_id": q.get("chunk_id", "unknown"),
                    "chunk_content": q.get("chunk_content", "")[:500],  # Limit chunk content size
                    "question_index": q.get("question_index", 0),
                    "generation_method": q.get("generation_method", "hyqe"),
                    "question_style": q.get("config", {}).get("style", "unknown"),
                    "question_quality": q.get("config", {}).get("quality", "unknown"),
                    "type": "question",  # Mark as question for identification
                    "created_at": str(time.time())  # Add timestamp
                }
                for q in batch
            ]
            
            # Validate metadata for ChromaDB compatibility
            batch_metadata = [Config.validate_metadata(meta) for meta in batch_metadata]
            
            # Add to collection
            collection.add(
                documents=batch_questions,
                ids=batch_ids,
                metadatas=batch_metadata
            )
            
            added_count += len(batch)
            
            # Update progress
            if progress_callback:
                progress = (i + len(batch)) / len(questions)
                progress_callback(progress)
            
            logger.debug(f"Added batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            logger.error(f"Failed to add batch {i//batch_size + 1}: {str(e)}")
            continue
    
    total_count = existing_count + added_count
    logger.info(f"Questions collection now contains {total_count} questions ({added_count} new)")
    
    return total_count

# Add the method to the class
DatabaseManager.populate_questions_collection_optimized = populate_questions_collection_optimized

# Test it
db_manager = DatabaseManager()
print(f"Has optimized method: {hasattr(db_manager, 'populate_questions_collection_optimized')}")

# Test with simple data
test_questions = [
    {
        'question': 'Test question?',
        'chunk_id': 'test_1',
        'chunk_content': 'This is test content.',
        'question_index': 0,
        'generation_method': 'hyqe',
        'config': {'style': 'natural', 'quality': 'balanced'}
    }
]

try:
    result = db_manager.populate_questions_collection_optimized(test_questions, skip_existing=False)
    print(f"Method executed successfully, result: {result}")
except Exception as e:
    print(f"Method failed: {e}")
    import traceback
    traceback.print_exc()
