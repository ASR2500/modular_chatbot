"""NER (Named Entity Recognition) configuration for the Python FAQ chatbot."""

import os
from typing import List, Dict, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NERConfig:
    """Configuration class for Named Entity Recognition."""
    
    # SpaCy Model Settings
    SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
    
    # Entity Types to Extract (SpaCy default labels)
    # See: https://spacy.io/api/annotation#named-entities
    DEFAULT_ENTITY_TYPES = {
        "PERSON",      # People, including fictional
        "ORG",         # Companies, agencies, institutions
        "GPE",         # Countries, cities, states
        "LANGUAGE",    # Named languages
        "PRODUCT",     # Objects, vehicles, foods, etc.
        "EVENT",       # Named hurricanes, battles, wars, sports events
        "WORK_OF_ART", # Titles of books, songs, etc.
        "LAW",         # Named documents made into laws
        "DATE",        # Absolute or relative dates or periods
        "TIME",        # Times smaller than a day
        "PERCENT",     # Percentage, including "%"
        "MONEY",       # Monetary values, including unit
        "QUANTITY",    # Measurements, as of weight or distance
        "ORDINAL",     # "first", "second", etc.
        "CARDINAL",    # Numerals that do not fall under another type
    }
    
    # Python-specific entities to focus on
    PYTHON_FOCUSED_ENTITIES = {
        "ORG",         # Python Software Foundation, Django, Flask, etc.
        "PRODUCT",     # Python packages, libraries, frameworks
        "LANGUAGE",    # Python, JavaScript, C++, etc.
        "PERSON",      # Guido van Rossum, etc.
        "EVENT",       # PyCon, conferences, etc.
        "WORK_OF_ART", # Books, documentation, etc.
        "DATE",        # Python version release dates
        "CARDINAL",    # Version numbers, counts
        "ORDINAL",     # Python 2, Python 3, etc.
    }
    
    # Entity types to extract (configurable via environment)
    ENTITY_TYPES_TO_EXTRACT = set(os.getenv("NER_ENTITY_TYPES", ",".join(PYTHON_FOCUSED_ENTITIES)).split(","))
    
    # Confidence threshold for entity extraction
    CONFIDENCE_THRESHOLD = float(os.getenv("NER_CONFIDENCE_THRESHOLD", "0.5"))
    
    # Custom entity patterns for Python-specific terms
    PYTHON_ENTITY_PATTERNS = [
        # Python versions
        {"label": "PYTHON_VERSION", "pattern": [
            {"TEXT": {"REGEX": r"Python\s*[0-9]\.[0-9]+(\.[0-9]+)?"}},
        ]},
        # Python packages/libraries
        {"label": "PYTHON_PACKAGE", "pattern": [
            {"TEXT": {"IN": ["pip", "conda", "virtualenv", "venv", "poetry", "pipenv"]}},
        ]},
        # Python frameworks
        {"label": "PYTHON_FRAMEWORK", "pattern": [
            {"TEXT": {"IN": ["Django", "Flask", "FastAPI", "Tornado", "Pyramid", "Bottle"]}},
        ]},
        # Python data science libraries
        {"label": "PYTHON_LIBRARY", "pattern": [
            {"TEXT": {"IN": ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "tensorflow", "pytorch"]}},
        ]},
        # Python keywords/concepts
        {"label": "PYTHON_CONCEPT", "pattern": [
            {"TEXT": {"IN": ["lambda", "decorator", "generator", "iterator", "comprehension", "metaclass"]}},
        ]},
    ]
    
    # Minimum entity length to consider
    MIN_ENTITY_LENGTH = int(os.getenv("NER_MIN_ENTITY_LENGTH", "2"))
    
    # Maximum number of entities to extract per text
    MAX_ENTITIES_PER_TEXT = int(os.getenv("NER_MAX_ENTITIES", "20"))
    
    # Whether to include entity positions
    INCLUDE_POSITIONS = os.getenv("NER_INCLUDE_POSITIONS", "true").lower() == "true"
    
    # Whether to normalize entity text
    NORMALIZE_ENTITIES = os.getenv("NER_NORMALIZE_ENTITIES", "true").lower() == "true"
    
    # Stop words to exclude from entities
    ENTITY_STOP_WORDS = {
        "python", "code", "example", "function", "method", "class", "variable",
        "string", "number", "list", "dict", "tuple", "set", "file", "data",
        "program", "script", "module", "package", "library", "framework",
        "error", "exception", "bug", "issue", "problem", "solution", "answer",
        "question", "help", "support", "documentation", "tutorial", "guide"
    }
    
    @classmethod
    def get_entity_types(cls) -> Set[str]:
        """Get the set of entity types to extract."""
        return cls.ENTITY_TYPES_TO_EXTRACT
    
    @classmethod
    def is_valid_entity(cls, entity_text: str, entity_label: str) -> bool:
        """Check if an entity is valid based on configuration rules.
        
        Args:
            entity_text: The text of the entity
            entity_label: The label/type of the entity
            
        Returns:
            True if the entity is valid, False otherwise
        """
        # Check minimum length
        if len(entity_text) < cls.MIN_ENTITY_LENGTH:
            return False
        
        # Check if entity type is in our extraction list
        if entity_label not in cls.ENTITY_TYPES_TO_EXTRACT:
            return False
        
        # Check if entity is a stop word (case-insensitive)
        if cls.NORMALIZE_ENTITIES and entity_text.lower() in cls.ENTITY_STOP_WORDS:
            return False
        
        return True
    
    @classmethod
    def normalize_entity_text(cls, text: str) -> str:
        """Normalize entity text if normalization is enabled.
        
        Args:
            text: The entity text to normalize
            
        Returns:
            Normalized entity text
        """
        if not cls.NORMALIZE_ENTITIES:
            return text
        
        # Basic normalization: strip whitespace, handle common cases
        normalized = text.strip()
        
        # Handle Python-specific normalizations
        if normalized.lower().startswith("python"):
            # Normalize Python version mentions
            import re
            normalized = re.sub(r"python\s*([0-9])", r"Python \1", normalized, flags=re.IGNORECASE)
        
        return normalized
    
    @classmethod
    def get_display_config(cls) -> Dict[str, any]:
        """Get configuration for display purposes.
        
        Returns:
            Dictionary with display-friendly configuration
        """
        return {
            "spacy_model": cls.SPACY_MODEL,
            "entity_types_count": len(cls.ENTITY_TYPES_TO_EXTRACT),
            "entity_types": sorted(list(cls.ENTITY_TYPES_TO_EXTRACT)),
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "min_entity_length": cls.MIN_ENTITY_LENGTH,
            "max_entities_per_text": cls.MAX_ENTITIES_PER_TEXT,
            "include_positions": cls.INCLUDE_POSITIONS,
            "normalize_entities": cls.NORMALIZE_ENTITIES,
            "custom_patterns_count": len(cls.PYTHON_ENTITY_PATTERNS),
            "stop_words_count": len(cls.ENTITY_STOP_WORDS)
        }
