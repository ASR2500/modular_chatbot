"""HyQE (Hypothetical Question Embeddings) configuration for the Python FAQ chatbot."""

import os
from typing import List, Dict, Set, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HyQEConfig:
    """Configuration class for HyQE (Hypothetical Question Embeddings)."""
    
    # OpenAI Settings for HyQE
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HYQE_MODEL = os.getenv("HYQE_MODEL", "gpt-3.5-turbo")
    
    # HyQE Core Settings
    ENABLE_HYQE = os.getenv("ENABLE_HYQE", "true").lower() == "true"
    
    # Number of questions to generate per chunk
    QUESTIONS_PER_CHUNK = int(os.getenv("HYQE_QUESTIONS_PER_CHUNK", "3"))
    
    # Question style - Controls the type of questions generated
    QUESTION_STYLE = os.getenv("HYQE_QUESTION_STYLE", "natural")
    QUESTION_STYLES = {
        "natural": "Generate natural, conversational questions that users might ask",
        "instructional": "Generate instructional questions focusing on how-to and procedural knowledge",
        "faq": "Generate FAQ-style questions that are commonly asked",
        "exam-style": "Generate exam-style questions that test understanding and knowledge"
    }
    
    # Quality level - Controls the specificity and depth of questions
    QUALITY = os.getenv("HYQE_QUALITY", "balanced")
    QUALITY_LEVELS = {
        "vague": "Generate general, open-ended questions that explore broad concepts",
        "specific": "Generate precise, detail-oriented questions that target specific information and edge cases",
        "balanced": "Generate clear and useful questions covering the main ideas with appropriate specificity"
    }
    
    # Temperature - Controls creativity in question generation
    TEMPERATURE = float(os.getenv("HYQE_TEMPERATURE", "0.4"))
    
    # Max tokens - Length limit for question generation
    MAX_TOKENS = int(os.getenv("HYQE_MAX_TOKENS", "100"))
    
    # Additional options
    PREPEND_SYSTEM_PROMPT = os.getenv("HYQE_PREPEND_SYSTEM_PROMPT", "true").lower() == "true"
    BATCH_MODE = os.getenv("HYQE_BATCH_MODE", "true").lower() == "true"  # Enable batch processing by default
    BATCH_SIZE = int(os.getenv("HYQE_BATCH_SIZE", "5"))  # Number of chunks to process in one batch
    MAX_CONCURRENT_REQUESTS = int(os.getenv("HYQE_MAX_CONCURRENT_REQUESTS", "3"))  # Limit concurrent API calls
    INCLUDE_CHUNK_SUMMARY = os.getenv("HYQE_INCLUDE_CHUNK_SUMMARY", "true").lower() == "true"
    FALLBACK_TO_CHUNK_TEXT = os.getenv("HYQE_FALLBACK_TO_CHUNK_TEXT", "true").lower() == "true"
    VERBOSE = os.getenv("HYQE_VERBOSE", "false").lower() == "true"
    
    # Content change detection
    ENABLE_CONTENT_HASHING = os.getenv("HYQE_ENABLE_CONTENT_HASHING", "true").lower() == "true"
    SKIP_UNCHANGED_CONTENT = os.getenv("HYQE_SKIP_UNCHANGED_CONTENT", "true").lower() == "true"
    
    # Caching for performance
    ENABLE_CACHE = os.getenv("HYQE_ENABLE_CACHE", "true").lower() == "true"
    CACHE_SIZE = int(os.getenv("HYQE_CACHE_SIZE", "200"))
    
    # Embedding settings
    EMBEDDING_MODEL = os.getenv("HYQE_EMBEDDING_MODEL", "text-embedding-3-small")
    COLLECTION_NAME_SUFFIX = "_questions"
    
    # Quality control settings
    MIN_QUESTION_LENGTH = int(os.getenv("HYQE_MIN_QUESTION_LENGTH", "10"))
    MAX_QUESTION_LENGTH = int(os.getenv("HYQE_MAX_QUESTION_LENGTH", "200"))
    DUPLICATE_THRESHOLD = float(os.getenv("HYQE_DUPLICATE_THRESHOLD", "0.85"))
    
    # Default prompts for different quality levels
    QUALITY_PROMPTS = {
        "vague": """Generate general, open-ended questions that explore broad concepts and encourage discussion. 
Focus on conceptual understanding rather than specific details. Use phrases like "What are the implications of...", 
"How might...", "What factors influence...". Questions should be thought-provoking and invite exploration.""",
        
        "specific": """Generate precise, detail-oriented questions that target specific information, edge cases, and technical details. 
Focus on concrete facts, specific parameters, exact procedures, and particular scenarios. Use phrases like "What exactly happens when...", 
"Which specific parameter...", "In what exact circumstances...". Questions should be technically focused and specific.""",
        
        "balanced": """Generate clear and useful questions covering the main ideas with appropriate specificity. 
Balance between conceptual understanding and practical details. Mix general concepts with specific examples. 
Questions should be practical, actionable, and cover both the 'what' and 'how' aspects of the content."""
    }
    
    # Style-specific prompt modifications
    STYLE_PROMPTS = {
        "natural": """Generate natural, conversational questions that real users might ask in everyday situations. 
Use informal language and common phrasing. Questions should feel like they come from someone learning or curious about the topic.""",
        
        "instructional": """Generate instructional questions focusing on how-to procedures, step-by-step processes, and practical implementation. 
Use action-oriented language with phrases like "How do I...", "What steps...", "How can I...". Focus on practical application.""",
        
        "faq": """Generate FAQ-style questions that are commonly asked about this topic. 
Use clear, direct language typical of documentation. Questions should address common concerns, misconceptions, and practical needs.""",
        
        "exam-style": """Generate exam-style questions that test understanding, knowledge retention, and problem-solving ability. 
Use academic language and testing formats. Include questions that assess comprehension, application, and analysis."""
    }
    
    # Default system prompts
    DEFAULT_SYSTEM_PROMPTS = {
        "python": "You are a Python expert who understands what questions developers commonly ask about Python programming.",
        "programming": "You are a programming expert who understands what questions developers commonly ask about software development.",
        "general": "You are a helpful assistant who understands what questions users commonly ask about various topics."
    }
    
    @classmethod
    def validate(cls):
        """Validate HyQE configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for HyQE")
        
        if cls.QUESTION_STYLE not in cls.QUESTION_STYLES:
            raise ValueError(f"Invalid question style: {cls.QUESTION_STYLE}. Must be one of: {list(cls.QUESTION_STYLES.keys())}")
        
        if cls.QUALITY not in cls.QUALITY_LEVELS:
            raise ValueError(f"Invalid quality level: {cls.QUALITY}. Must be one of: {list(cls.QUALITY_LEVELS.keys())}")
        
        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 1:
            raise ValueError("HYQE_TEMPERATURE must be between 0 and 1")
        
        if cls.MAX_TOKENS < 10 or cls.MAX_TOKENS > 500:
            raise ValueError("HYQE_MAX_TOKENS must be between 10 and 500")
        
        if cls.QUESTIONS_PER_CHUNK < 1 or cls.QUESTIONS_PER_CHUNK > 10:
            raise ValueError("HYQE_QUESTIONS_PER_CHUNK must be between 1 and 10")
        
        if cls.DUPLICATE_THRESHOLD < 0 or cls.DUPLICATE_THRESHOLD > 1:
            raise ValueError("HYQE_DUPLICATE_THRESHOLD must be between 0 and 1")
    
    @classmethod
    def get_prompt_template(cls, domain: str = "python") -> str:
        """Get the prompt template for question generation.
        
        Args:
            domain: Domain type (python, programming, general)
            
        Returns:
            Prompt template string
        """
        # Start with base prompt
        base_prompt = f"""Given the following text content, generate {cls.QUESTIONS_PER_CHUNK} relevant questions that users might ask about this information.

Content:
{{content}}

Instructions:
- {cls.STYLE_PROMPTS.get(cls.QUESTION_STYLE, cls.STYLE_PROMPTS["natural"])}
- {cls.QUALITY_PROMPTS.get(cls.QUALITY, cls.QUALITY_PROMPTS["balanced"])}
- Each question should be on a new line
- Questions should be diverse and cover different aspects of the content
- Avoid repetitive or overly similar questions
- Focus on the most important and useful information

Generate {cls.QUESTIONS_PER_CHUNK} questions:"""
        
        # Add chunk summary instruction if enabled
        if cls.INCLUDE_CHUNK_SUMMARY:
            base_prompt = base_prompt.replace("Content:", "Content Summary: {{summary}}\n\nFull Content:")
        
        return base_prompt
    
    @classmethod
    def get_system_prompt(cls, domain: str = "python") -> str:
        """Get the system prompt for question generation.
        
        Args:
            domain: Domain type
            
        Returns:
            System prompt string
        """
        if cls.PREPEND_SYSTEM_PROMPT:
            return cls.DEFAULT_SYSTEM_PROMPTS.get(domain, cls.DEFAULT_SYSTEM_PROMPTS["general"])
        else:
            return "You are a helpful assistant."
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, any]:
        """Get configuration as dictionary for logging/debugging."""
        return {
            "enabled": cls.ENABLE_HYQE,
            "model": cls.HYQE_MODEL,
            "questions_per_chunk": cls.QUESTIONS_PER_CHUNK,
            "question_style": cls.QUESTION_STYLE,
            "quality": cls.QUALITY,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
            "prepend_system_prompt": cls.PREPEND_SYSTEM_PROMPT,
            "batch_mode": cls.BATCH_MODE,
            "include_chunk_summary": cls.INCLUDE_CHUNK_SUMMARY,
            "fallback_to_chunk_text": cls.FALLBACK_TO_CHUNK_TEXT,
            "verbose": cls.VERBOSE,
            "cache_enabled": cls.ENABLE_CACHE,
            "embedding_model": cls.EMBEDDING_MODEL
        }
    
    @classmethod
    def get_quality_description(cls, quality: str) -> str:
        """Get description for a quality level."""
        return cls.QUALITY_LEVELS.get(quality, "Unknown quality level")
    
    @classmethod
    def get_style_description(cls, style: str) -> str:
        """Get description for a question style."""
        return cls.QUESTION_STYLES.get(style, "Unknown question style")
