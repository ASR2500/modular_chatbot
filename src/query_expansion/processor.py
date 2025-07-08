"""Query Expansion processor for generating alternative queries."""

import asyncio
import logging
from typing import List, Dict, Set, Optional, Tuple
from functools import lru_cache
import hashlib
import json
import re
from difflib import SequenceMatcher

import openai
from openai import OpenAI

from src.query_expansion_config import QueryExpansionConfig

logger = logging.getLogger(__name__)

class QueryExpansionProcessor:
    """Processor for expanding queries to improve retrieval."""
    
    def __init__(self, config: Optional[QueryExpansionConfig] = None):
        """Initialize the query expansion processor.
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or QueryExpansionConfig()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        self._cache = {} if self.config.ENABLE_CACHE else None
        logger.info("Query expansion processor initialized")
    
    def expand_query(self, query: str, num_expansions: Optional[int] = None, 
                    strategies: Optional[Set[str]] = None) -> List[str]:
        """Expand a query into multiple alternative queries.
        
        Args:
            query: The original query to expand
            num_expansions: Number of expansions to generate
            strategies: Set of strategies to use for expansion
            
        Returns:
            List of expanded queries
        """
        if not query or not query.strip():
            return []
        
        query = query.strip()
        num_expansions = num_expansions or self.config.DEFAULT_NUM_EXPANSIONS
        strategies = strategies or self.config.get_expansion_strategies()
        
        # Check cache first
        if self._cache is not None:
            cache_key = self._get_cache_key(query, num_expansions, strategies)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for query: {query}")
                return self._cache[cache_key]
        
        try:
            # Generate expansions using OpenAI
            expansions = self._generate_expansions(query, num_expansions, strategies)
            
            # Filter and validate expansions
            valid_expansions = self._filter_expansions(expansions, query)
            
            # Cache the result
            if self._cache is not None:
                self._update_cache(cache_key, valid_expansions)
            
            logger.info(f"Generated {len(valid_expansions)} expansions for query: {query}")
            return valid_expansions
            
        except Exception as e:
            logger.error(f"Error expanding query '{query}': {str(e)}")
            return []
    
    def _generate_expansions(self, query: str, num_expansions: int, strategies: Set[str]) -> List[str]:
        """Generate expansions using OpenAI.
        
        Args:
            query: The original query
            num_expansions: Number of expansions to generate
            strategies: Set of strategies to use
            
        Returns:
            List of generated expansions
        """
        system_prompt = self.config.get_system_prompt(query, num_expansions, strategies)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.QUERY_EXPANSION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Expand this query: {query}"}
                ],
                temperature=self.config.EXPANSION_TEMPERATURE,
                max_tokens=self.config.MAX_EXPANSION_TOKENS,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            
            content = response.choices[0].message.content
            if not content:
                return []
            
            # Parse expansions (assuming one per line)
            expansions = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Remove any numbering or bullet points
            cleaned_expansions = []
            for expansion in expansions:
                # Remove common prefixes like "1.", "•", "-", etc.
                cleaned = re.sub(r'^\d+\.\s*', '', expansion)
                cleaned = re.sub(r'^[•\-\*]\s*', '', cleaned)
                cleaned = cleaned.strip()
                if cleaned:
                    cleaned_expansions.append(cleaned)
            
            return cleaned_expansions
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return []
    
    def _filter_expansions(self, expansions: List[str], original_query: str) -> List[str]:
        """Filter expansions based on quality and similarity.
        
        Args:
            expansions: List of generated expansions
            original_query: The original query
            
        Returns:
            List of filtered expansions
        """
        valid_expansions = []
        seen_expansions = set()
        
        for expansion in expansions:
            # Basic validation
            if not self.config.is_valid_expansion(expansion, original_query):
                continue
            
            # Check for duplicates (case-insensitive)
            expansion_lower = expansion.lower()
            if expansion_lower in seen_expansions:
                continue
            
            # Check similarity to original and other expansions
            if self._is_too_similar(expansion, original_query):
                continue
            
            # Check similarity to other valid expansions
            if any(self._is_too_similar(expansion, valid_exp) for valid_exp in valid_expansions):
                continue
            
            valid_expansions.append(expansion)
            seen_expansions.add(expansion_lower)
        
        return valid_expansions
    
    def _is_too_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are too similar.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if texts are too similar, False otherwise
        """
        similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity > self.config.SIMILARITY_THRESHOLD
    
    def _get_cache_key(self, query: str, num_expansions: int, strategies: Set[str]) -> str:
        """Generate a cache key for the query and parameters.
        
        Args:
            query: The query
            num_expansions: Number of expansions
            strategies: Set of strategies
            
        Returns:
            Cache key string
        """
        key_data = {
            'query': query,
            'num_expansions': num_expansions,
            'strategies': sorted(list(strategies)),
            'temperature': self.config.EXPANSION_TEMPERATURE
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _update_cache(self, cache_key: str, expansions: List[str]):
        """Update the cache with new expansions.
        
        Args:
            cache_key: The cache key
            expansions: The expansions to cache
        """
        if self._cache is None:
            return
        
        # Remove oldest entries if cache is full
        if len(self._cache) >= self.config.CACHE_SIZE:
            # Remove the first (oldest) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = expansions
    
    def get_expansion_suggestions(self, query: str) -> Dict[str, List[str]]:
        """Get expansion suggestions organized by strategy.
        
        Args:
            query: The query to expand
            
        Returns:
            Dictionary with strategies as keys and expansions as values
        """
        suggestions = {}
        
        # Try each strategy individually
        for strategy in self.config.EXPANSION_STRATEGIES.keys():
            try:
                strategy_expansions = self.expand_query(
                    query, 
                    num_expansions=2,  # Fewer per strategy
                    strategies={strategy}
                )
                if strategy_expansions:
                    suggestions[strategy] = strategy_expansions
            except Exception as e:
                logger.warning(f"Failed to generate expansions for strategy '{strategy}': {str(e)}")
        
        return suggestions
    
    def expand_with_python_context(self, query: str, num_expansions: Optional[int] = None) -> List[str]:
        """Expand query with Python-specific context.
        
        Args:
            query: The original query
            num_expansions: Number of expansions to generate
            
        Returns:
            List of Python-focused expansions
        """
        python_strategies = {"python_specific", "context"}
        return self.expand_query(query, num_expansions, python_strategies)
    
    def clear_cache(self):
        """Clear the expansion cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Query expansion cache cleared")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if self._cache is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self._cache),
            "max_size": self.config.CACHE_SIZE,
            "hit_ratio": "N/A"  # Would need to track hits/misses
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get processor statistics.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            "model": self.config.QUERY_EXPANSION_MODEL,
            "temperature": self.config.EXPANSION_TEMPERATURE,
            "max_tokens": self.config.MAX_EXPANSION_TOKENS,
            "strategies": list(self.config.get_expansion_strategies()),
            "cache_stats": self.get_cache_stats(),
            "config": self.config.get_display_config()
        }

# Async version for future use
class AsyncQueryExpansionProcessor(QueryExpansionProcessor):
    """Async version of the query expansion processor."""
    
    async def expand_query_async(self, query: str, num_expansions: Optional[int] = None, 
                               strategies: Optional[Set[str]] = None) -> List[str]:
        """Async version of expand_query.
        
        Args:
            query: The original query to expand
            num_expansions: Number of expansions to generate
            strategies: Set of strategies to use for expansion
            
        Returns:
            List of expanded queries
        """
        # Run the synchronous version in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.expand_query, 
            query, 
            num_expansions, 
            strategies
        )
