"""Test script for the NER processor."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ner.processor import NERProcessor, Entity

def test_ner_processor():
    """Test the NER processor with sample text."""
    print("🧪 Testing NER Processor")
    
    try:
        processor = NERProcessor()
        print("✅ NER Processor initialized successfully")
        
        # Get processor stats
        stats = processor.get_processor_stats()
        print(f"✅ Model loaded: {stats['model_loaded']}")
        print(f"✅ Model name: {stats['model_name']}")
        print(f"✅ Pipeline components: {stats.get('pipeline_components', 'N/A')}")
        
        sample_texts = [
            "Python is a programming language created by Guido van Rossum.",
            "I'm using Django and Flask to build web applications.",
            "NumPy and pandas are essential for data science in Python.",
            "The Python Software Foundation organizes PyCon every year.",
            "Python 3.9 was released on October 5, 2020."
        ]
        
        all_entities = []
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n📝 Text {i}: {text}")
            entities = processor.extract_entities(text)
            print(f"🔍 Found {len(entities)} entities:")
            
            for entity in entities:
                print(f"  - {entity.text} ({entity.label}): {entity.description}")
                all_entities.append(entity)
            
            # Show highlighted version
            if entities:
                highlighted = processor.highlight_entities_in_text(text, entities)
                print(f"✨ Highlighted: {highlighted}")
        
        # Show summary
        print(f"\n📊 Summary:")
        summary = processor.get_entity_summary(all_entities)
        print(f"  - Total entities: {summary['total']}")
        print(f"  - Unique texts: {summary['unique_texts']}")
        print(f"  - Entity types: {summary['types']}")
        print(f"  - By type: {summary['by_type']}")
        
        # Show most common entities
        print(f"\n🏆 Most Common Entities:")
        common_entities = processor.get_most_common_entities(all_entities, top_n=5)
        for text, label, count in common_entities:
            print(f"  - {text} ({label}): {count} times")
        
        print("\n🎉 NER Processor test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ner_processor()
    
    if success:
        print("\n✅ All NER tests passed!")
    else:
        print("\n❌ NER tests failed. Please check the error messages above.")
