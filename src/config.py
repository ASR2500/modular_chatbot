"""Configuration module for the Python FAQ RAG chatbot."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # ChromaDB Settings
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "python_faq_collection")
    
    # Data Settings
    DATA_PATH = os.getenv("DATA_PATH", "./data/pythonfaq.csv")
    
    # RAG Settings
    DEFAULT_N_RESULTS = int(os.getenv("DEFAULT_N_RESULTS", "3"))
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
    
    # Query Expansion Settings
    ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    QUERY_EXPANSION_NUM_QUERIES = int(os.getenv("QUERY_EXPANSION_NUM_QUERIES", "3"))
    
    # Generation Settings
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Streamlit Settings
    PAGE_TITLE = "Python FAQ Chatbot"
    PAGE_ICON = "🐍"
    
    # HyDE Settings
    ENABLE_HYDE = os.getenv("ENABLE_HYDE", "true").lower() == "true"
    HYDE_TEMPERATURE = float(os.getenv("HYDE_TEMPERATURE", "0.2"))
    HYDE_MAX_TOKENS = int(os.getenv("HYDE_MAX_TOKENS", "150"))
    HYDE_ANSWER_STYLE = os.getenv("HYDE_ANSWER_STYLE", "concise")
    HYDE_INCLUDE_EXAMPLES = os.getenv("HYDE_INCLUDE_EXAMPLES", "true").lower() == "true"
    HYDE_BATCH_MODE = os.getenv("HYDE_BATCH_MODE", "false").lower() == "true"
    HYDE_BATCH_SIZE = int(os.getenv("HYDE_BATCH_SIZE", "3"))

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if not os.path.exists(cls.DATA_PATH):
            raise ValueError(f"Data file not found: {cls.DATA_PATH}")
    
    @classmethod
    def get_system_prompt(cls):
        """Get the system prompt for the LLM."""
        return """You are a Python expert assistant with deep knowledge of Python programming. 
Your role is to provide accurate, helpful, and comprehensive answers to Python-related questions.

Guidelines:
1. Always base your answers on the provided context from the Python FAQ
2. If the context doesn't contain sufficient information, acknowledge this and provide what you can
3. Use clear, concise language and provide examples when helpful
4. Structure your responses logically with proper formatting
5. When referencing context, be specific about which source you're using
6. If asked about topics not covered in the context, politely explain the limitation
7. Be precise and avoid speculation beyond what's in the context
8. Format code snippets properly and explain them clearly"""
    
    @classmethod
    def validate_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize metadata for ChromaDB compatibility.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Sanitized metadata dictionary
        """
        sanitized = {}
        
        for key, value in metadata.items():
            # ChromaDB only accepts str, int, float, bool, or None
            if isinstance(value, (str, int, float, bool, type(None))):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                sanitized[key] = ', '.join(str(item) for item in value)
            elif isinstance(value, dict):
                # Convert dicts to JSON strings
                import json
                sanitized[key] = json.dumps(value)
            else:
                # Convert other types to strings
                sanitized[key] = str(value)
        
        return sanitized
