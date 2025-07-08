"""HyDE (Hypothetical Document Embeddings) processor for generating hypothetical documents."""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import hashlib
import json
import re
import time

import openai
from openai import OpenAI

from src.hyde_config import HyDEConfig

logger = logging.getLogger(__name__)

class HyDEProcessor:
    """Processor for generating hypothetical documents using HyDE technique."""
    
    def __init__(self, config: Optional[HyDEConfig] = None):
        """Initialize the HyDE processor.
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or HyDEConfig()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        
        # Initialize cache if enabled
        if self.config.ENABLE_CACHE:
            self._cache = {}
            self._cache_stats = {"hits": 0, "misses": 0}
        else:
            self._cache = None
            self._cache_stats = None
        
        # Validate configuration
        self.config.validate()
        
        logger.info(f"HyDE processor initialized with config: {self.config.get_config_dict()}")
    
    def generate_hypothetical_document(self, query: str, domain: str = "python") -> str:
        """Generate a single hypothetical document for a query.
        
        Args:
            query: The user query
            domain: Domain context (python, programming, general)
            
        Returns:
            Generated hypothetical document
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to HyDE processor")
            return ""
        
        query = query.strip()
        
        # Check cache first
        if self._cache is not None:
            cache_key = self._get_cache_key(query, domain)
            if cache_key in self._cache:
                self._cache_stats["hits"] += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return self._cache[cache_key]
            else:
                self._cache_stats["misses"] += 1
        
        try:
            # Generate hypothetical document
            hypothetical_doc = self._generate_document(query, domain)
            
            # Cache the result
            if self._cache is not None:
                cache_key = self._get_cache_key(query, domain)
                self._cache[cache_key] = hypothetical_doc
                
                # Maintain cache size
                if len(self._cache) > self.config.CACHE_SIZE:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
            
            logger.debug(f"Generated hypothetical document for query: {query[:50]}...")
            return hypothetical_doc
            
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {str(e)}")
            return ""
    
    def generate_hypothetical_documents(self, query: str, domain: str = "python") -> List[str]:
        """Generate multiple hypothetical documents for a query (batch mode).
        
        Args:
            query: The user query
            domain: Domain context (python, programming, general)
            
        Returns:
            List of generated hypothetical documents
        """
        if not self.config.BATCH_MODE:
            # Single document generation
            doc = self.generate_hypothetical_document(query, domain)
            return [doc] if doc else []
        
        # Batch generation
        documents = []
        for i in range(self.config.BATCH_SIZE):
            try:
                # Slightly vary the prompt for diversity
                varied_query = self._add_variation(query, i)
                doc = self.generate_hypothetical_document(varied_query, domain)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to generate document {i+1}/{self.config.BATCH_SIZE}: {str(e)}")
                continue
        
        logger.info(f"Generated {len(documents)} hypothetical documents for query: {query[:50]}...")
        return documents
    
    def _generate_document(self, query: str, domain: str) -> str:
        """Generate a hypothetical document using the LLM.
        
        Args:
            query: The user query
            domain: Domain context
            
        Returns:
            Generated hypothetical document
        """
        # Get the prompt template
        prompt_template = self.config.get_prompt_template(domain)
        prompt = prompt_template.format(query=query)
        
        # Add domain context if specified
        if self.config.DOMAIN_CONTEXT:
            system_prompt = self.config.DOMAIN_CONTEXT
        else:
            system_prompt = "You are a helpful assistant."
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.HYDE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            hypothetical_doc = response.choices[0].message.content
            
            # Post-process the document
            hypothetical_doc = self._post_process_document(hypothetical_doc)
            
            return hypothetical_doc
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def _post_process_document(self, document: str) -> str:
        """Post-process the generated document.
        
        Args:
            document: Raw generated document
            
        Returns:
            Processed document
        """
        if not document:
            return ""
        
        # Remove common prefixes that might be generated
        prefixes_to_remove = [
            "Answer:",
            "Response:",
            "Here's the answer:",
            "The answer is:",
            "Hypothetical answer:",
        ]
        
        for prefix in prefixes_to_remove:
            if document.strip().startswith(prefix):
                document = document.strip()[len(prefix):].strip()
                break
        
        # Clean up extra whitespace
        document = re.sub(r'\n\s*\n', '\n\n', document)
        document = document.strip()
        
        # Ensure minimum length
        if len(document) < 20:
            logger.warning(f"Generated document is too short: {len(document)} characters")
            return ""
        
        return document
    
    def _add_variation(self, query: str, variation_index: int) -> str:
        """Add slight variation to query for batch generation.
        
        Args:
            query: Original query
            variation_index: Index of variation (0-based)
            
        Returns:
            Slightly modified query
        """
        if variation_index == 0:
            return query
        
        # Simple variations to encourage diversity
        variations = [
            f"Can you explain {query}",
            f"What is {query}",
            f"How does {query} work",
            f"Tell me about {query}",
            f"Describe {query}"
        ]
        
        # Use modulo to cycle through variations
        variation_template = variations[variation_index % len(variations)]
        
        # If query is already a question, use it as-is
        if query.strip().endswith('?'):
            return query
        
        return variation_template
    
    def _get_cache_key(self, query: str, domain: str) -> str:
        """Generate cache key for a query and domain.
        
        Args:
            query: The user query
            domain: Domain context
            
        Returns:
            Cache key string
        """
        # Include relevant config in cache key
        config_str = f"{self.config.TEMPERATURE}_{self.config.MAX_TOKENS}_{self.config.ANSWER_STYLE}_{self.config.INCLUDE_EXAMPLES}"
        cache_input = f"{query}_{domain}_{config_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get_cache_stats(self) -> Optional[Dict[str, int]]:
        """Get cache statistics.
        
        Returns:
            Cache statistics or None if caching disabled
        """
        if self._cache_stats is None:
            return None
        
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (self._cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 1),
            "cache_size": len(self._cache) if self._cache else 0
        }
    
    def clear_cache(self):
        """Clear the cache."""
        if self._cache is not None:
            self._cache.clear()
            self._cache_stats = {"hits": 0, "misses": 0}
            logger.info("HyDE cache cleared")
    
    def get_stats(self) -> Dict[str, any]:
        """Get processor statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "enabled": self.config.ENABLE_HYDE,
            "model": self.config.HYDE_MODEL,
            "config": self.config.get_config_dict()
        }
        
        # Add cache stats if available
        cache_stats = self.get_cache_stats()
        if cache_stats:
            stats["cache"] = cache_stats
        
        return stats
    
    def is_enabled(self) -> bool:
        """Check if HyDE is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return self.config.ENABLE_HYDE
