"""HyDE (Hypothetical Document Embeddings) configuration for the Python FAQ chatbot."""

import os
from typing import List, Dict, Set, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HyDEConfig:
    """Configuration class for HyDE (Hypothetical Document Embeddings)."""
    
    # OpenAI Settings for HyDE
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HYDE_MODEL = os.getenv("HYDE_MODEL", "gpt-4o-mini")
    
    # HyDE Core Settings
    ENABLE_HYDE = os.getenv("ENABLE_HYDE", "true").lower() == "true"
    
    # Temperature - Controls how deterministic vs. creative the answer is
    # 0.2-0.3 for factual; 0.7+ for brainstorming
    TEMPERATURE = float(os.getenv("HYDE_TEMPERATURE", "0.2"))
    
    # Max tokens - Length limit for fake answer
    # 100-250 for FAQs
    MAX_TOKENS = int(os.getenv("HYDE_MAX_TOKENS", "150"))
    
    # Answer style - Controls tone and depth
    ANSWER_STYLE = os.getenv("HYDE_ANSWER_STYLE", "concise")
    ANSWER_STYLES = {
        "concise": "Provide brief, to-the-point answers",
        "detailed": "Provide comprehensive, detailed explanations", 
        "steps": "Structure answers as step-by-step instructions",
        "bullets": "Format answers as bullet points"
    }
    
    # Domain context - Tells the LLM to bias toward a domain
    DOMAIN_CONTEXT = os.getenv("HYDE_DOMAIN_CONTEXT", "Answer as a Python expert")
    
    # Include examples - Toggle for sample code, equations, etc.
    INCLUDE_EXAMPLES = os.getenv("HYDE_INCLUDE_EXAMPLES", "true").lower() == "true"
    
    # Batch mode - Enables multiple HyDE generations (e.g., for ensemble embedding)
    BATCH_MODE = os.getenv("HYDE_BATCH_MODE", "false").lower() == "true"
    BATCH_SIZE = int(os.getenv("HYDE_BATCH_SIZE", "3"))
    
    # Caching for performance
    ENABLE_CACHE = os.getenv("HYDE_ENABLE_CACHE", "true").lower() == "true"
    CACHE_SIZE = int(os.getenv("HYDE_CACHE_SIZE", "100"))
    
    # Default prompts for different domains
    DEFAULT_PROMPTS = {
        "python": """You are a Python expert. Given a question, generate a hypothetical answer that would be found in a Python FAQ or documentation.

Guidelines:
- Write as if you're answering from official Python documentation
- Include relevant Python concepts and terminology
- Keep the answer factual and informative
- Focus on practical information
- Use proper Python syntax in code examples if needed

Question: {query}

Generate a hypothetical answer:""",
        
        "programming": """You are a programming expert. Given a question, generate a hypothetical answer that would be found in programming documentation or tutorials.

Guidelines:
- Write as if you're answering from official programming documentation
- Include relevant programming concepts and terminology
- Keep the answer factual and informative
- Focus on practical information
- Use proper syntax in code examples if needed

Question: {query}

Generate a hypothetical answer:""",
        
        "general": """Given a question, generate a hypothetical answer that would be found in relevant documentation or FAQ.

Guidelines:
- Write as if you're answering from official documentation
- Include relevant concepts and terminology
- Keep the answer factual and informative
- Focus on practical information

Question: {query}

Generate a hypothetical answer:"""
    }
    
    # Style-specific prompt modifications
    STYLE_MODIFIERS = {
        "concise": "Keep the answer brief and to the point.",
        "detailed": "Provide a comprehensive and detailed explanation.",
        "steps": "Structure the answer as clear, numbered steps.",
        "bullets": "Format the answer as bullet points."
    }
    
    # Example inclusion prompts
    EXAMPLE_PROMPTS = {
        "python": "Include a simple Python code example if relevant.",
        "programming": "Include a simple code example if relevant.",
        "general": "Include a simple example if relevant."
    }
    
    @classmethod
    def validate(cls):
        """Validate HyDE configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for HyDE")
        
        if cls.ANSWER_STYLE not in cls.ANSWER_STYLES:
            raise ValueError(f"Invalid answer style: {cls.ANSWER_STYLE}. Must be one of: {list(cls.ANSWER_STYLES.keys())}")
        
        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 1:
            raise ValueError("HYDE_TEMPERATURE must be between 0 and 1")
        
        if cls.MAX_TOKENS < 10 or cls.MAX_TOKENS > 1000:
            raise ValueError("HYDE_MAX_TOKENS must be between 10 and 1000")
        
        if cls.BATCH_SIZE < 1 or cls.BATCH_SIZE > 10:
            raise ValueError("HYDE_BATCH_SIZE must be between 1 and 10")
    
    @classmethod
    def get_prompt_template(cls, domain: str = "python") -> str:
        """Get the prompt template for a specific domain.
        
        Args:
            domain: Domain type (python, programming, general)
            
        Returns:
            Prompt template string
        """
        base_prompt = cls.DEFAULT_PROMPTS.get(domain, cls.DEFAULT_PROMPTS["general"])
        
        # Add style modifier
        style_modifier = cls.STYLE_MODIFIERS.get(cls.ANSWER_STYLE, "")
        if style_modifier:
            base_prompt = base_prompt.replace("Generate a hypothetical answer:", 
                                           f"{style_modifier} Generate a hypothetical answer:")
        
        # Add example inclusion
        if cls.INCLUDE_EXAMPLES:
            example_prompt = cls.EXAMPLE_PROMPTS.get(domain, cls.EXAMPLE_PROMPTS["general"])
            base_prompt = base_prompt.replace("Generate a hypothetical answer:", 
                                           f"{example_prompt} Generate a hypothetical answer:")
        
        return base_prompt
    
    @classmethod
    def get_domain_context(cls) -> str:
        """Get the domain context string."""
        return cls.DOMAIN_CONTEXT
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, any]:
        """Get configuration as dictionary for logging/debugging."""
        return {
            "enabled": cls.ENABLE_HYDE,
            "model": cls.HYDE_MODEL,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
            "answer_style": cls.ANSWER_STYLE,
            "domain_context": cls.DOMAIN_CONTEXT,
            "include_examples": cls.INCLUDE_EXAMPLES,
            "batch_mode": cls.BATCH_MODE,
            "batch_size": cls.BATCH_SIZE if cls.BATCH_MODE else 1,
            "cache_enabled": cls.ENABLE_CACHE
        }
