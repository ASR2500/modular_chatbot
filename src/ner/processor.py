"""Named Entity Recognition module for the Python FAQ chatbot."""

import spacy
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import re

from src.ner_config import NERConfig

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents a named entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        return f"{self.text} ({self.label})"

class NERProcessor:
    """Named Entity Recognition processor using SpaCy."""
    
    def __init__(self, config: Optional[NERConfig] = None):
        """Initialize the NER processor.
        
        Args:
            config: NER configuration instance
        """
        self.config = config or NERConfig()
        self.nlp = None
        self._entity_descriptions = self._load_entity_descriptions()
        self._initialize_spacy()
    
    def _initialize_spacy(self):
        """Initialize SpaCy model with custom patterns."""
        try:
            self.nlp = spacy.load(self.config.SPACY_MODEL)
            logger.info(f"Loaded SpaCy model: {self.config.SPACY_MODEL}")
            
            # Add custom patterns if available
            if hasattr(self.config, 'PYTHON_ENTITY_PATTERNS') and self.config.PYTHON_ENTITY_PATTERNS:
                self._add_custom_patterns()
            
        except OSError as e:
            logger.error(f"Failed to load SpaCy model '{self.config.SPACY_MODEL}': {e}")
            logger.error("Please install the model using: python -m spacy download en_core_web_sm")
            raise
    
    def _add_custom_patterns(self):
        """Add custom entity patterns to the SpaCy pipeline."""
        try:
            # Add entity ruler for custom patterns
            if "entity_ruler" not in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                ruler.add_patterns(self.config.PYTHON_ENTITY_PATTERNS)
                logger.info(f"Added {len(self.config.PYTHON_ENTITY_PATTERNS)} custom entity patterns")
        except Exception as e:
            logger.warning(f"Failed to add custom patterns: {e}")
    
    def _load_entity_descriptions(self) -> Dict[str, str]:
        """Load entity type descriptions."""
        return {
            "PERSON": "People, including fictional characters",
            "ORG": "Companies, agencies, institutions, organizations",
            "GPE": "Countries, cities, states, geopolitical entities",
            "LANGUAGE": "Named languages",
            "PRODUCT": "Objects, vehicles, foods, software products",
            "EVENT": "Named events, conferences, battles, sports events",
            "WORK_OF_ART": "Titles of books, songs, movies, documentation",
            "LAW": "Named documents made into laws",
            "DATE": "Absolute or relative dates or periods",
            "TIME": "Times smaller than a day",
            "PERCENT": "Percentage values including '%'",
            "MONEY": "Monetary values including currency",
            "QUANTITY": "Measurements of weight, distance, etc.",
            "ORDINAL": "First, second, third, etc.",
            "CARDINAL": "Numerals that do not fall under another type",
            "PYTHON_VERSION": "Python version numbers",
            "PYTHON_PACKAGE": "Python packages and package managers",
            "PYTHON_FRAMEWORK": "Python web frameworks",
            "PYTHON_LIBRARY": "Python libraries and modules",
            "PYTHON_CONCEPT": "Python programming concepts"
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted entities
        """
        if not self.nlp:
            logger.warning("SpaCy model not initialized")
            return []
        
        if not text or not text.strip():
            return []
        
        try:
            # Process the text
            doc = self.nlp(text)
            
            # Extract entities
            entities = []
            seen_entities = set()  # To avoid duplicates
            
            for ent in doc.ents:
                # Check if entity meets our criteria
                if not self.config.is_valid_entity(ent.text, ent.label_):
                    continue
                
                # Normalize entity text
                normalized_text = self.config.normalize_entity_text(ent.text)
                
                # Avoid duplicates (case-insensitive)
                entity_key = (normalized_text.lower(), ent.label_)
                if entity_key in seen_entities:
                    continue
                seen_entities.add(entity_key)
                
                # Create entity object
                entity = Entity(
                    text=normalized_text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,  # SpaCy doesn't provide confidence scores by default
                    description=self._entity_descriptions.get(ent.label_, "")
                )
                
                entities.append(entity)
                
                # Limit number of entities
                if len(entities) >= self.config.MAX_ENTITIES_PER_TEXT:
                    break
            
            logger.debug(f"Extracted {len(entities)} entities from text of length {len(text)}")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def extract_entities_from_query_and_context(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, List[Entity]]:
        """Extract entities from both query and retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved contexts from RAG
            
        Returns:
            Dictionary with 'query' and 'contexts' entities
        """
        result = {
            "query": self.extract_entities(query),
            "contexts": []
        }
        
        # Extract entities from contexts
        for i, context in enumerate(contexts):
            context_entities = []
            
            # Extract from question
            if "question" in context.get("metadata", {}):
                question_entities = self.extract_entities(context["metadata"]["question"])
                context_entities.extend(question_entities)
            
            # Extract from answer
            if "answer" in context.get("metadata", {}):
                answer_entities = self.extract_entities(context["metadata"]["answer"])
                context_entities.extend(answer_entities)
            
            result["contexts"].append({
                "context_index": i,
                "entities": context_entities
            })
        
        return result
    
    def get_entity_summary(self, entities: List[Entity]) -> Dict[str, Any]:
        """Get summary statistics for extracted entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Summary statistics
        """
        if not entities:
            return {"total": 0, "by_type": {}, "unique_texts": 0}
        
        # Count by type
        by_type = {}
        unique_texts = set()
        
        for entity in entities:
            by_type[entity.label] = by_type.get(entity.label, 0) + 1
            unique_texts.add(entity.text.lower())
        
        return {
            "total": len(entities),
            "by_type": by_type,
            "unique_texts": len(unique_texts),
            "types": sorted(list(by_type.keys()))
        }
    
    def filter_entities_by_type(self, entities: List[Entity], entity_types: Set[str]) -> List[Entity]:
        """Filter entities by their types.
        
        Args:
            entities: List of entities to filter
            entity_types: Set of entity types to keep
            
        Returns:
            Filtered list of entities
        """
        return [entity for entity in entities if entity.label in entity_types]
    
    def get_most_common_entities(self, entities: List[Entity], top_n: int = 10) -> List[Tuple[str, str, int]]:
        """Get the most common entities.
        
        Args:
            entities: List of entities
            top_n: Number of top entities to return
            
        Returns:
            List of (text, label, count) tuples
        """
        entity_counts = {}
        
        for entity in entities:
            key = (entity.text.lower(), entity.label)
            entity_counts[key] = entity_counts.get(key, 0) + 1
        
        # Sort by count (descending)
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [(text, label, count) for (text, label), count in sorted_entities[:top_n]]
    
    def highlight_entities_in_text(self, text: str, entities: List[Entity]) -> str:
        """Highlight entities in text with markdown formatting.
        
        Args:
            text: Original text
            entities: List of entities to highlight
            
        Returns:
            Text with highlighted entities
        """
        if not entities:
            return text
        
        # Sort entities by start position (descending) to avoid position shifts
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
        
        highlighted_text = text
        
        for entity in sorted_entities:
            # Create highlighted version
            highlighted = f"**{entity.text}** *({entity.label})*"
            
            # Replace in text
            highlighted_text = (
                highlighted_text[:entity.start] +
                highlighted +
                highlighted_text[entity.end:]
            )
        
        return highlighted_text
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get statistics about the NER processor.
        
        Returns:
            Dictionary with processor statistics
        """
        stats = {
            "model_loaded": self.nlp is not None,
            "model_name": self.config.SPACY_MODEL,
            "config": self.config.get_display_config()
        }
        
        if self.nlp:
            stats.update({
                "pipeline_components": list(self.nlp.pipe_names),
                "has_custom_patterns": "entity_ruler" in self.nlp.pipe_names,
                "entity_types_supported": len(self._entity_descriptions)
            })
        
        return stats

# Standalone function for quick testing
def test_ner_processor():
    """Test the NER processor with sample text."""
    processor = NERProcessor()
    
    sample_texts = [
        "Python is a programming language created by Guido van Rossum.",
        "I'm using Django and Flask to build web applications.",
        "NumPy and pandas are essential for data science in Python.",
        "The Python Software Foundation organizes PyCon every year.",
        "Python 3.9 was released on October 5, 2020."
    ]
    
    for text in sample_texts:
        print(f"\nText: {text}")
        entities = processor.extract_entities(text)
        for entity in entities:
            print(f"  - {entity}")
        
        # Show highlighted version
        highlighted = processor.highlight_entities_in_text(text, entities)
        print(f"Highlighted: {highlighted}")

if __name__ == "__main__":
    test_ner_processor()
