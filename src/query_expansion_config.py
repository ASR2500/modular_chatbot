"""Query Expansion configuration for the Python FAQ chatbot."""

import os
from typing import List, Dict, Set, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QueryExpansionConfig:
    """Configuration class for Query Expansion."""
    
    # OpenAI Settings for query expansion
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QUERY_EXPANSION_MODEL = os.getenv("QUERY_EXPANSION_MODEL", "gpt-4o-mini")
    
    # Query Expansion Settings
    DEFAULT_NUM_EXPANSIONS = int(os.getenv("QE_NUM_EXPANSIONS", "3"))
    MAX_EXPANSIONS = int(os.getenv("QE_MAX_EXPANSIONS", "10"))
    MIN_EXPANSIONS = int(os.getenv("QE_MIN_EXPANSIONS", "1"))
    
    # Temperature for query expansion (lower = more conservative)
    EXPANSION_TEMPERATURE = float(os.getenv("QE_TEMPERATURE", "0.7"))
    
    # Maximum tokens for expansion generation
    MAX_EXPANSION_TOKENS = int(os.getenv("QE_MAX_TOKENS", "500"))
    
    # Query expansion strategies
    EXPANSION_STRATEGIES = {
        "synonyms": "Generate synonymous queries using different terminology",
        "specificity": "Create more specific versions of the query",
        "generalization": "Create more general versions of the query", 
        "context": "Add relevant context to the query",
        "python_specific": "Transform into Python-specific variations"
    }
    
    # Default strategy combination
    DEFAULT_STRATEGIES = set(os.getenv("QE_STRATEGIES", "synonyms,specificity,python_specific").split(","))
    
    # Python-specific expansion keywords
    PYTHON_CONTEXT_KEYWORDS = {
        "programming": ["coding", "development", "scripting"],
        "function": ["method", "procedure", "callable"],
        "library": ["package", "module", "framework"],
        "error": ["exception", "bug", "issue"],
        "data": ["information", "dataset", "structure"],
        "loop": ["iteration", "cycle", "repeat"],
        "variable": ["identifier", "name", "value"],
        "class": ["object", "type", "blueprint"],
        "string": ["text", "character sequence", "str"],
        "list": ["array", "sequence", "collection"]
    }
    
    # Quality filters
    MIN_EXPANSION_LENGTH = int(os.getenv("QE_MIN_LENGTH", "5"))
    MAX_EXPANSION_LENGTH = int(os.getenv("QE_MAX_LENGTH", "200"))
    
    # Similarity threshold to avoid too similar expansions
    SIMILARITY_THRESHOLD = float(os.getenv("QE_SIMILARITY_THRESHOLD", "0.8"))
    
    # Whether to enable query expansion by default
    ENABLE_BY_DEFAULT = os.getenv("QE_ENABLE_DEFAULT", "true").lower() == "true"
    
    # Cache settings
    ENABLE_CACHE = os.getenv("QE_ENABLE_CACHE", "true").lower() == "true"
    CACHE_SIZE = int(os.getenv("QE_CACHE_SIZE", "100"))
    
    # System prompt for query expansion
    EXPANSION_SYSTEM_PROMPT = """You are a query expansion specialist for a Python FAQ chatbot. 
Your job is to generate alternative versions of user queries to improve information retrieval.

Guidelines:
1. Generate {num_expansions} alternative queries for the given input
2. Each expansion should maintain the original intent while using different wording
3. Include Python-specific terminology when relevant
4. Make some queries more specific and others more general
5. Use synonyms and related terms
6. Keep expansions natural and readable
7. Focus on variations that would find different relevant documents
8. Return only the expanded queries, one per line
9. Do not include explanations or numbering

Strategies to use: {strategies}

Original query: {query}

Generate {num_expansions} expanded queries:"""
    
    @classmethod
    def get_expansion_strategies(cls) -> Set[str]:
        """Get the set of expansion strategies to use."""
        return cls.DEFAULT_STRATEGIES
    
    @classmethod
    def get_system_prompt(cls, query: str, num_expansions: int, strategies: Optional[Set[str]] = None) -> str:
        """Get the system prompt for query expansion.
        
        Args:
            query: The original query
            num_expansions: Number of expansions to generate
            strategies: Set of strategies to use
            
        Returns:
            Formatted system prompt
        """
        if strategies is None:
            strategies = cls.DEFAULT_STRATEGIES
        
        strategies_desc = ", ".join([cls.EXPANSION_STRATEGIES.get(s, s) for s in strategies])
        
        return cls.EXPANSION_SYSTEM_PROMPT.format(
            num_expansions=num_expansions,
            strategies=strategies_desc,
            query=query
        )
    
    @classmethod
    def is_valid_expansion(cls, expansion: str, original_query: str) -> bool:
        """Check if an expansion is valid.
        
        Args:
            expansion: The expanded query
            original_query: The original query
            
        Returns:
            True if the expansion is valid, False otherwise
        """
        # Check length constraints
        if len(expansion) < cls.MIN_EXPANSION_LENGTH or len(expansion) > cls.MAX_EXPANSION_LENGTH:
            return False
        
        # Check if it's too similar to original (basic check)
        if expansion.lower().strip() == original_query.lower().strip():
            return False
        
        # Check for empty or whitespace-only expansions
        if not expansion.strip():
            return False
        
        return True
    
    @classmethod
    def get_python_synonyms(cls, text: str) -> List[str]:
        """Get Python-specific synonyms for terms in the text.
        
        Args:
            text: The text to find synonyms for
            
        Returns:
            List of synonym suggestions
        """
        synonyms = []
        text_lower = text.lower()
        
        for term, alternatives in cls.PYTHON_CONTEXT_KEYWORDS.items():
            if term in text_lower:
                synonyms.extend(alternatives)
        
        return synonyms
    
    @classmethod
    def get_display_config(cls) -> Dict[str, any]:
        """Get configuration for display purposes.
        
        Returns:
            Dictionary with display-friendly configuration
        """
        return {
            "model": cls.QUERY_EXPANSION_MODEL,
            "default_num_expansions": cls.DEFAULT_NUM_EXPANSIONS,
            "max_expansions": cls.MAX_EXPANSIONS,
            "min_expansions": cls.MIN_EXPANSIONS,
            "temperature": cls.EXPANSION_TEMPERATURE,
            "strategies": sorted(list(cls.DEFAULT_STRATEGIES)),
            "available_strategies": sorted(list(cls.EXPANSION_STRATEGIES.keys())),
            "enable_by_default": cls.ENABLE_BY_DEFAULT,
            "enable_cache": cls.ENABLE_CACHE,
            "cache_size": cls.CACHE_SIZE,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD
        }
    
    @classmethod
    def validate(cls):
        """Validate the configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for query expansion")
        
        if cls.DEFAULT_NUM_EXPANSIONS < cls.MIN_EXPANSIONS or cls.DEFAULT_NUM_EXPANSIONS > cls.MAX_EXPANSIONS:
            raise ValueError(f"DEFAULT_NUM_EXPANSIONS must be between {cls.MIN_EXPANSIONS} and {cls.MAX_EXPANSIONS}")
        
        if cls.EXPANSION_TEMPERATURE < 0.0 or cls.EXPANSION_TEMPERATURE > 2.0:
            raise ValueError("EXPANSION_TEMPERATURE must be between 0.0 and 2.0")
