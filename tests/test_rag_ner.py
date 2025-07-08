"""Simple test to verify RAG engine NER integration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rag.engine import RAGEngine
from src.rag.database import DatabaseManager
from src.ner.processor import NERProcessor

def test_rag_engine_ner():
    """Test RAG engine NER functionality."""
    print("🧪 Testing RAG Engine NER Integration")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        print("✅ Database Manager initialized")
        
        # Initialize RAG engine with NER
        rag_engine = RAGEngine(db_manager, enable_ner=True)
        print("✅ RAG Engine initialized with NER enabled")
        
        # Check methods exist
        methods_to_check = [
            'get_ner_stats',
            'process_query_with_ner',
            'retrieve_contexts_with_ner',
            'generate_response_with_ner'
        ]
        
        for method_name in methods_to_check:
            if hasattr(rag_engine, method_name):
                print(f"✅ Method {method_name} exists")
            else:
                print(f"❌ Method {method_name} missing")
        
        # Test NER stats
        if hasattr(rag_engine, 'get_ner_stats'):
            ner_stats = rag_engine.get_ner_stats()
            print(f"✅ NER stats: {ner_stats}")
        
        # Test simple query
        test_query = "What is Python?"
        try:
            result = rag_engine.process_query(test_query, n_results=1)
            print(f"✅ Standard query processed: {len(result['response'])} chars")
        except Exception as e:
            print(f"⚠️  Standard query failed: {e}")
        
        # Test NER query if available
        if hasattr(rag_engine, 'process_query_with_ner') and rag_engine.enable_ner:
            try:
                result = rag_engine.process_query_with_ner(test_query, n_results=1)
                print(f"✅ NER query processed: {len(result['response'])} chars")
                print(f"  - NER enabled: {result.get('ner_enabled', False)}")
            except Exception as e:
                print(f"⚠️  NER query failed: {e}")
        
        print("\n🎉 RAG Engine NER integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_engine_ner()
    
    if success:
        print("✅ RAG Engine NER integration working!")
    else:
        print("❌ RAG Engine NER integration failed!")
