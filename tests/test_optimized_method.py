#!/usr/bin/env python3
"""Test script to validate the database manager optimized method."""

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

# Now import
from src.rag.database import DatabaseManager

# Test
db_manager = DatabaseManager()
print(f"DatabaseManager created: {db_manager}")
print(f"Has optimized method: {hasattr(db_manager, 'populate_questions_collection_optimized')}")

if hasattr(db_manager, 'populate_questions_collection_optimized'):
    print("Method is available!")
    
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
else:
    print("Method is NOT available")
    print(f"Available methods: {[m for m in dir(db_manager) if not m.startswith('_')]}")
