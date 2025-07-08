#!/usr/bin/env python3
"""Test script to verify HyQE data flow and UI display."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.database import DatabaseManager
from src.rag.engine import RAGEngine
from src.data.processor import DataProcessor
from src.config import Config

def test_hyqe_data_flow():
    """Test the HyQE data flow from database to UI."""
    
    print("🔍 Testing HyQE data flow...")
    
    # Initialize components
    db_manager = DatabaseManager()
    rag_engine = RAGEngine(database_manager=db_manager, enable_hyqe=True)
    
    # Test query
    test_query = "How do I create a list in Python?"
    
    print(f"📝 Testing query: {test_query}")
    
    # Test retrieval with HyQE
    contexts = rag_engine.retrieve_contexts(
        query=test_query,
        n_results=5,
        use_hyqe=True
    )
    
    print(f"📊 Retrieved {len(contexts)} contexts")
    
    # Check for question-based contexts
    question_contexts = [ctx for ctx in contexts if ctx.get("source") == "question_embedding"]
    document_contexts = [ctx for ctx in contexts if ctx.get("source") == "document_embedding"]
    
    print(f"❓ Question-based contexts: {len(question_contexts)}")
    print(f"📄 Document-based contexts: {len(document_contexts)}")
    
    # Examine question context structure
    if question_contexts:
        print("\n🔍 Examining first question context:")
        ctx = question_contexts[0]
        print(f"Source: {ctx.get('source')}")
        print(f"Similarity: {ctx.get('similarity', 0):.3f}")
        
        metadata = ctx.get("metadata", {})
        print(f"Metadata keys: {list(metadata.keys())}")
        print(f"Full metadata: {metadata}")
        
        source_question = metadata.get("source_question", "MISSING")
        chunk_content = metadata.get("chunk_content", "MISSING")
        
        print(f"Source question: {source_question[:100]}..." if len(source_question) > 100 else f"Source question: {source_question}")
        print(f"Chunk content: {chunk_content[:100]}..." if len(chunk_content) > 100 else f"Chunk content: {chunk_content}")
        
        # Check alternative fields
        alt_content = metadata.get("source_content_preview", "MISSING")
        print(f"Alternative content (source_content_preview): {alt_content[:100]}..." if len(alt_content) > 100 else f"Alternative content: {alt_content}")
        
        # Check if data is available
        if source_question == "MISSING":
            print("❌ Source question is missing!")
        else:
            print("✅ Source question is available")
            
        if chunk_content == "MISSING" and alt_content == "MISSING":
            print("❌ No chunk content available!")
        else:
            print("✅ Some chunk content is available")
    else:
        print("❌ No question-based contexts found!")
        
        # Check if questions collection exists and has data
        questions_collection = db_manager.get_or_create_questions_collection()
        questions_count = questions_collection.count()
        print(f"📊 Questions collection has {questions_count} questions")
        
        if questions_count == 0:
            print("❌ Questions collection is empty! HyQE not populated.")
        else:
            print("✅ Questions collection has data")
            
            # Try a direct query to questions collection
            direct_results = db_manager.query_questions_collection(test_query, 3)
            print(f"📋 Direct question query returned {len(direct_results)} results")
            
            if direct_results:
                print("Sample question result:")
                result = direct_results[0]
                print(f"  Question: {result.get('question', 'N/A')}")
                print(f"  Metadata: {result.get('metadata', {})}")
                print(f"  Similarity: {result.get('similarity', 0):.3f}")
                
                # Check how this gets processed in query_both_collections
                print("\n🔧 Testing query_both_collections processing...")
                both_results = db_manager.query_both_collections(test_query, 3)
                if both_results:
                    print(f"  Both collections returned {len(both_results)} results")
                    question_results = [r for r in both_results if r.get("source") == "question_embedding"]
                    print(f"  Question results: {len(question_results)}")
                    if question_results:
                        qr = question_results[0]
                        print(f"  Processed question result metadata: {qr.get('metadata', {})}")
                        print(f"  Source question: {qr.get('metadata', {}).get('source_question', 'MISSING')}")
                        print(f"  Chunk content: {qr.get('metadata', {}).get('chunk_content', 'MISSING')}")

if __name__ == "__main__":
    test_hyqe_data_flow()
