"""HyQE (Hypothetical Question Embeddings) processor for generating hypothetical questions."""

import logging
from typing import List, Dict, Optional, Tuple, Any
from functools import lru_cache
import hashlib
import re
from difflib import SequenceMatcher
import numpy as np
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI

from src.hyqe_config import HyQEConfig

logger = logging.getLogger(__name__)

class HyQEProcessor:
    """Processor for generating hypothetical questions using HyQE technique."""
    
    def __init__(self, config: Optional[HyQEConfig] = None):
        """Initialize the HyQE processor.
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or HyQEConfig()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        
        # Initialize cache if enabled
        if self.config.ENABLE_CACHE:
            self._cache = {}
            self._cache_stats = {"hits": 0, "misses": 0}
        else:
            self._cache = None
            self._cache_stats = None
        
        # Content hash store for change detection
        self._content_hashes = {}
        
        # Statistics tracking
        self._generation_stats = {
            "total_chunks_processed": 0,
            "total_questions_generated": 0,
            "total_questions_filtered": 0,
            "total_api_calls": 0,
            "total_api_tokens": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_operations": 0,
            "average_questions_per_chunk": 0.0
        }
        
        # Batch processing settings
        self.batch_size = getattr(self.config, 'BATCH_SIZE', 5)
        self.max_concurrent_requests = getattr(self.config, 'MAX_CONCURRENT_REQUESTS', 3)
        
        # Validate configuration
        self.config.validate()
        
        logger.info(f"HyQE processor initialized with config: {self.config.get_config_dict()}")
    
    def generate_questions_for_chunks(self, chunks: List[Dict[str, Any]], 
                                    domain: str = "python",
                                    progress_callback=None) -> List[Dict[str, Any]]:
        """Generate hypothetical questions for multiple chunks with optimization.
        
        Args:
            chunks: List of chunk dictionaries with 'content' and optional 'id' keys
            domain: Domain context (python, programming, general)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of all generated questions with metadata
        """
        if not chunks:
            return []
        
        start_time = time.time()
        
        # Filter chunks that need processing (skip unchanged content)
        chunks_to_process = self._filter_chunks_for_processing(chunks)
        
        if not chunks_to_process:
            logger.info("No new chunks to process - all content unchanged")
            return self._get_cached_questions_for_chunks(chunks)
        
        logger.info(f"Processing {len(chunks_to_process)} new/changed chunks out of {len(chunks)} total")
        
        # Process chunks in batches for efficiency
        if self.config.BATCH_MODE and len(chunks_to_process) > 1:
            all_questions = self._process_chunks_in_batches(chunks_to_process, domain, progress_callback)
        else:
            all_questions = self._process_chunks_individually(chunks_to_process, domain, progress_callback)
        
        # Combine with cached results
        cached_questions = self._get_cached_questions_for_chunks([c for c in chunks if c not in chunks_to_process])
        all_questions.extend(cached_questions)
        
        # Remove duplicates across all questions
        all_questions = self._remove_global_duplicates(all_questions)
        
        # Update processing time
        processing_time = time.time() - start_time
        self._generation_stats["total_processing_time"] += processing_time
        
        logger.info(f"Generated {len(all_questions)} unique questions for {len(chunks)} chunks in {processing_time:.2f}s")
        
        return all_questions
    
    def _filter_chunks_for_processing(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter chunks to only process those that have changed or are new.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks that need processing
        """
        chunks_to_process = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id', '')
            
            # Generate content hash
            content_hash = self._generate_content_hash(content)
            
            # Check if content has changed
            if chunk_id not in self._content_hashes or self._content_hashes[chunk_id] != content_hash:
                chunks_to_process.append(chunk)
                self._content_hashes[chunk_id] = content_hash
        
        return chunks_to_process
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for content to detect changes.
        
        Args:
            content: Content string
            
        Returns:
            Content hash
        """
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cached_questions_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get cached questions for chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of cached questions
        """
        cached_questions = []
        
        if self._cache is None:
            return cached_questions
        
        for chunk in chunks:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id', '')
            
            cache_key = self._get_cache_key(content, "python")  # Default domain
            if cache_key in self._cache:
                questions = self._cache[cache_key]
                # Update chunk_id in cached questions
                for q in questions:
                    q['chunk_id'] = chunk_id
                cached_questions.extend(questions)
                self._cache_stats["hits"] += 1
        
        return cached_questions
    
    def _process_chunks_in_batches(self, chunks: List[Dict[str, Any]], 
                                  domain: str, progress_callback=None) -> List[Dict[str, Any]]:
        """Process chunks in batches for efficiency.
        
        Args:
            chunks: List of chunks to process
            domain: Domain context
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of generated questions
        """
        all_questions = []
        
        # Split chunks into batches
        batches = [chunks[i:i+self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        
        logger.info(f"Processing {len(chunks)} chunks in {len(batches)} batches")
        
        for batch_idx, batch in enumerate(batches):
            try:
                # Process batch
                batch_questions = self._process_batch(batch, domain)
                all_questions.extend(batch_questions)
                
                # Update statistics
                self._generation_stats["batch_operations"] += 1
                
                # Update progress
                if progress_callback:
                    progress = (batch_idx + 1) / len(batches)
                    progress_callback(progress)
                
                # Log progress
                if self.config.VERBOSE:
                    logger.info(f"Processed batch {batch_idx + 1}/{len(batches)} with {len(batch_questions)} questions")
                    
            except Exception as e:
                logger.warning(f"Failed to process batch {batch_idx}: {str(e)}")
                # Fall back to individual processing for this batch
                batch_questions = self._process_chunks_individually(batch, domain)
                all_questions.extend(batch_questions)
        
        return all_questions
    
    def _process_batch(self, chunks: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """Process a batch of chunks together for efficiency.
        
        Args:
            chunks: List of chunks in the batch
            domain: Domain context
            
        Returns:
            List of generated questions
        """
        # Combine chunk contents for batch processing
        combined_content = self._combine_chunks_for_batch(chunks)
        
        # Generate questions for the combined content
        questions = self._generate_questions_batch(combined_content, domain)
        
        # Process and distribute questions back to chunks
        batch_questions = self._distribute_questions_to_chunks(questions, chunks)
        
        return batch_questions
    
    def _combine_chunks_for_batch(self, chunks: List[Dict[str, Any]]) -> str:
        """Combine multiple chunks into a single content for batch processing.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Combined content string
        """
        combined_parts = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '').strip()
            if content:
                combined_parts.append(f"Content {i+1}:\n{content}")
        
        return "\n\n".join(combined_parts)
    
    def _generate_questions_batch(self, combined_content: str, domain: str) -> List[str]:
        """Generate questions for combined content in batch mode.
        
        Args:
            combined_content: Combined content from multiple chunks
            domain: Domain context
            
        Returns:
            List of generated questions
        """
        # Calculate total questions needed
        total_questions = len(combined_content.split("Content ")) * self.config.QUESTIONS_PER_CHUNK
        
        # Prepare the batch prompt
        prompt_template = self.config.get_prompt_template(domain)
        
        # Modify prompt for batch processing
        batch_prompt = prompt_template.replace(
            f"generate {self.config.QUESTIONS_PER_CHUNK} relevant questions",
            f"generate {total_questions} relevant questions"
        )
        
        prompt = batch_prompt.format(content=combined_content)
        system_prompt = self.config.get_system_prompt(domain)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.HYQE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.MAX_TOKENS * 2,  # Increase token limit for batch
                temperature=self.config.TEMPERATURE,
                presence_penalty=0.1,
                frequency_penalty=0.2
            )
            
            raw_questions = response.choices[0].message.content
            questions = self._parse_questions_from_response(raw_questions)
            
            # Update statistics
            self._generation_stats["total_api_calls"] += 1
            self._generation_stats["total_api_tokens"] += response.usage.total_tokens
            
            return questions
            
        except Exception as e:
            logger.error(f"Error in batch API call: {str(e)}")
            raise
    
    def _distribute_questions_to_chunks(self, questions: List[str], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute generated questions back to their respective chunks.
        
        Args:
            questions: List of generated questions
            chunks: List of original chunks
            
        Returns:
            List of question dictionaries with chunk associations
        """
        batch_questions = []
        questions_per_chunk = self.config.QUESTIONS_PER_CHUNK
        
        for i, chunk in enumerate(chunks):
            start_idx = i * questions_per_chunk
            end_idx = min(start_idx + questions_per_chunk, len(questions))
            
            chunk_questions = questions[start_idx:end_idx]
            chunk_id = chunk.get('id', f"chunk_{i}")
            content = chunk.get('content', '')
            
            for j, question in enumerate(chunk_questions):
                question_dict = {
                    "question": question,
                    "chunk_id": chunk_id,
                    "chunk_content": content,
                    "question_index": j,
                    "generation_method": "hyqe_batch",
                    "config": {
                        "style": self.config.QUESTION_STYLE,
                        "quality": self.config.QUALITY,
                        "temperature": self.config.TEMPERATURE
                    }
                }
                batch_questions.append(question_dict)
        
        return batch_questions
    
    def _process_chunks_individually(self, chunks: List[Dict[str, Any]], 
                                   domain: str, progress_callback=None) -> List[Dict[str, Any]]:
        """Process chunks individually (fallback method).
        
        Args:
            chunks: List of chunks to process
            domain: Domain context
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of generated questions
        """
        all_questions = []
        
        for i, chunk in enumerate(chunks):
            try:
                content = chunk.get('content', '')
                chunk_id = chunk.get('id', f"chunk_{i}")
                
                chunk_questions = self.generate_questions_for_chunk(content, chunk_id, domain)
                all_questions.extend(chunk_questions)
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(chunks)
                    progress_callback(progress)
                
                if self.config.VERBOSE and (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks individually")
                    
            except Exception as e:
                logger.warning(f"Failed to generate questions for chunk {i}: {str(e)}")
                continue
        
        return all_questions

    def generate_questions_for_chunk(self, content: str, chunk_id: str = None, 
                                   domain: str = "python") -> List[Dict[str, Any]]:
        """Generate hypothetical questions for a single chunk of content.
        
        Args:
            content: The content chunk to generate questions for
            chunk_id: Optional identifier for the chunk
            domain: Domain context (python, programming, general)
            
        Returns:
            List of question dictionaries with metadata
        """
        if not content or not content.strip():
            logger.warning("Empty content provided to HyQE processor")
            return []
        
        content = content.strip()
        
        # Track processing time
        start_time = time.time()
        
        # Generate content hash for tracking
        content_hash = self._generate_content_hash(content)
        chunk_id = chunk_id or "default"
        self._content_hashes[chunk_id] = content_hash
        
        # Check cache first
        if self._cache is not None:
            cache_key = self._get_cache_key(content, domain)
            if cache_key in self._cache:
                self._cache_stats["hits"] += 1
                self._generation_stats["cache_hits"] += 1
                if self.config.VERBOSE:
                    logger.debug(f"Cache hit for content: {content[:50]}...")
                return self._cache[cache_key]
            else:
                self._cache_stats["misses"] += 1
                self._generation_stats["cache_misses"] += 1
        
        try:
            # Generate questions
            questions = self._generate_questions(content, domain)
            
            # Process and filter questions
            processed_questions = self._process_questions(questions, content, chunk_id)
            
            # Cache the result
            if self._cache is not None:
                cache_key = self._get_cache_key(content, domain)
                self._cache[cache_key] = processed_questions
                
                # Maintain cache size
                if len(self._cache) > self.config.CACHE_SIZE:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
            
            # Update statistics
            self._update_generation_stats(1, len(processed_questions))
            
            # Update processing time
            processing_time = time.time() - start_time
            self._generation_stats["total_processing_time"] += processing_time
            
            if self.config.VERBOSE:
                logger.info(f"Generated {len(processed_questions)} questions for chunk: {content[:50]}...")
            
            return processed_questions
            
        except Exception as e:
            logger.error(f"Error generating questions for chunk: {str(e)}")
            if self.config.FALLBACK_TO_CHUNK_TEXT:
                # Return a default question based on the content
                return self._generate_fallback_questions(content, chunk_id)
            return []

    def _generate_questions(self, content: str, domain: str) -> List[str]:
        """Generate questions using the LLM.
        
        Args:
            content: The content to generate questions for
            domain: Domain context
            
        Returns:
            List of generated questions
        """
        # Prepare the prompt
        prompt_template = self.config.get_prompt_template(domain)
        
        # Add summary if enabled
        if self.config.INCLUDE_CHUNK_SUMMARY:
            summary = self._generate_content_summary(content)
            prompt = prompt_template.format(content=content, summary=summary)
        else:
            prompt = prompt_template.format(content=content)
        
        # Get system prompt
        system_prompt = self.config.get_system_prompt(domain)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.HYQE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                presence_penalty=0.1,
                frequency_penalty=0.2  # Slightly higher to encourage diverse questions
            )
            
            raw_questions = response.choices[0].message.content
            
            # Update statistics
            self._generation_stats["total_api_calls"] += 1
            if hasattr(response, 'usage') and response.usage:
                self._generation_stats["total_api_tokens"] += response.usage.total_tokens
            
            # Parse questions from the response
            questions = self._parse_questions_from_response(raw_questions)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def _generate_content_summary(self, content: str) -> str:
        """Generate a brief summary of the content.
        
        Args:
            content: Content to summarize
            
        Returns:
            Brief summary
        """
        # Simple extractive summary - take first sentence or first 100 characters
        sentences = content.split('.')
        if len(sentences) > 0 and len(sentences[0]) > 20:
            return sentences[0].strip() + "."
        
        # Fallback to first 100 characters
        return content[:100].strip() + "..." if len(content) > 100 else content.strip()
    
    def _parse_questions_from_response(self, raw_response: str) -> List[str]:
        """Parse questions from the LLM response.
        
        Args:
            raw_response: Raw response from the LLM
            
        Returns:
            List of parsed questions
        """
        if not raw_response:
            return []
        
        # Split by lines and clean up
        lines = raw_response.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove numbering (1., 2., etc.)
            line = re.sub(r'^\d+\.?\s*', '', line)
            
            # Remove bullet points
            line = re.sub(r'^[-*•]\s*', '', line)
            
            # Remove quotes
            line = line.strip('"\'')
            
            # Skip if too short or too long
            if len(line) < self.config.MIN_QUESTION_LENGTH or len(line) > self.config.MAX_QUESTION_LENGTH:
                continue
            
            # Ensure it ends with a question mark
            if not line.endswith('?'):
                line += '?'
            
            questions.append(line)
        
        return questions
    
    def _process_questions(self, questions: List[str], content: str, chunk_id: str) -> List[Dict[str, Any]]:
        """Process and filter generated questions.
        
        Args:
            questions: List of generated questions
            content: Original content
            chunk_id: Chunk identifier
            
        Returns:
            List of processed question dictionaries
        """
        processed_questions = []
        
        for i, question in enumerate(questions):
            # Create question dictionary
            question_dict = {
                "question": question,
                "chunk_id": chunk_id,
                "chunk_content": content,
                "question_index": i,
                "generation_method": "hyqe",
                "config": {
                    "style": self.config.QUESTION_STYLE,
                    "quality": self.config.QUALITY,
                    "temperature": self.config.TEMPERATURE
                }
            }
            
            processed_questions.append(question_dict)
        
        # Remove duplicates within this chunk
        processed_questions = self._remove_duplicates(processed_questions)
        
        # Update filter statistics
        filtered_count = len(questions) - len(processed_questions)
        self._generation_stats["total_questions_filtered"] += filtered_count
        
        return processed_questions
    
    def _remove_duplicates(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate questions based on similarity.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            List of unique questions
        """
        if len(questions) <= 1:
            return questions
        
        unique_questions = []
        
        for question in questions:
            is_duplicate = False
            question_text = question["question"].lower()
            
            for existing in unique_questions:
                existing_text = existing["question"].lower()
                
                # Check similarity
                similarity = SequenceMatcher(None, question_text, existing_text).ratio()
                
                if similarity > self.config.DUPLICATE_THRESHOLD:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_questions.append(question)
        
        return unique_questions
    
    def _remove_global_duplicates(self, all_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates across all questions.
        
        Args:
            all_questions: List of all question dictionaries
            
        Returns:
            List of globally unique questions
        """
        return self._remove_duplicates(all_questions)
    
    def _generate_fallback_questions(self, content: str, chunk_id: str) -> List[Dict[str, Any]]:
        """Generate fallback questions when main generation fails.
        
        Args:
            content: Content to generate questions for
            chunk_id: Chunk identifier
            
        Returns:
            List of fallback questions
        """
        # Simple fallback questions based on content analysis
        fallback_questions = []
        
        # Basic question templates
        templates = [
            "What is the main topic discussed in this content?",
            "How does this information relate to Python programming?",
            "What are the key points mentioned here?"
        ]
        
        for i, template in enumerate(templates):
            if i < self.config.QUESTIONS_PER_CHUNK:
                question_dict = {
                    "question": template,
                    "chunk_id": chunk_id,
                    "chunk_content": content,
                    "question_index": i,
                    "generation_method": "fallback",
                    "config": {
                        "style": "fallback",
                        "quality": "basic",
                        "temperature": 0.0
                    }
                }
                fallback_questions.append(question_dict)
        
        return fallback_questions
    
    def _get_cache_key(self, content: str, domain: str) -> str:
        """Generate cache key for content and domain.
        
        Args:
            content: Content string
            domain: Domain context
            
        Returns:
            Cache key string
        """
        # Include relevant config in cache key
        config_str = f"{self.config.QUESTION_STYLE}_{self.config.QUALITY}_{self.config.TEMPERATURE}_{self.config.QUESTIONS_PER_CHUNK}"
        cache_input = f"{content[:200]}_{domain}_{config_str}"  # Use first 200 chars to avoid huge keys
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _update_generation_stats(self, chunks_processed: int, questions_generated: int):
        """Update generation statistics.
        
        Args:
            chunks_processed: Number of chunks processed
            questions_generated: Number of questions generated
        """
        self._generation_stats["total_chunks_processed"] += chunks_processed
        self._generation_stats["total_questions_generated"] += questions_generated
        
        # Calculate average
        if self._generation_stats["total_chunks_processed"] > 0:
            self._generation_stats["average_questions_per_chunk"] = (
                self._generation_stats["total_questions_generated"] / 
                self._generation_stats["total_chunks_processed"]
            )
    
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
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics.
        
        Returns:
            Generation statistics
        """
        return self._generation_stats.copy()
    
    def save_content_hashes(self, filepath: str = None):
        """Save content hashes to disk for persistence.
        
        Args:
            filepath: Optional path to save hashes
        """
        if not self.config.ENABLE_CONTENT_HASHING:
            return
        
        if filepath is None:
            filepath = "hyqe_content_hashes.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self._content_hashes, f, indent=2)
            logger.info(f"Saved {len(self._content_hashes)} content hashes to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save content hashes: {str(e)}")
    
    def load_content_hashes(self, filepath: str = None):
        """Load content hashes from disk for persistence.
        
        Args:
            filepath: Optional path to load hashes from
        """
        if not self.config.ENABLE_CONTENT_HASHING:
            return
        
        if filepath is None:
            filepath = "hyqe_content_hashes.json"
        
        try:
            with open(filepath, 'r') as f:
                self._content_hashes = json.load(f)
            logger.info(f"Loaded {len(self._content_hashes)} content hashes from {filepath}")
        except FileNotFoundError:
            logger.info("No existing content hashes found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load content hashes: {str(e)}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current stats.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        stats = self._generation_stats
        
        # Check cache effectiveness
        if self._cache_stats and self._cache_stats["misses"] > 0:
            hit_rate = self._cache_stats["hits"] / (self._cache_stats["hits"] + self._cache_stats["misses"])
            if hit_rate < 0.3:
                recommendations.append("Cache hit rate is low. Consider increasing cache size or enabling content hashing.")
        
        # Check batch processing
        if stats["total_chunks_processed"] > 10 and stats["batch_operations"] == 0:
            recommendations.append("Consider enabling batch processing for better efficiency with large datasets.")
        
        # Check API usage
        if stats["total_api_calls"] > 50:
            avg_tokens_per_call = stats["total_api_tokens"] / stats["total_api_calls"]
            if avg_tokens_per_call < 50:
                recommendations.append("API calls are using few tokens. Consider increasing batch size.")
        
        # Check processing time
        if stats["total_processing_time"] > 60:  # More than 1 minute
            recommendations.append("Processing time is high. Consider enabling caching and batch processing.")
        
        return recommendations
    
    def clear_cache(self):
        """Clear the question generation cache."""
        if self._cache is not None:
            self._cache.clear()
            if self._cache_stats:
                self._cache_stats = {"hits": 0, "misses": 0}
            logger.info("HyQE cache cleared")
    
    def get_cache_efficiency(self) -> float:
        """Get cache efficiency as a percentage.
        
        Returns:
            Cache hit rate as percentage (0-100)
        """
        if not self._cache_stats:
            return 0.0
        
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        if total_requests == 0:
            return 0.0
        
        return (self._cache_stats["hits"] / total_requests) * 100
    
    def estimate_cost_savings(self) -> Dict[str, float]:
        """Estimate cost savings from optimizations.
        
        Returns:
            Dictionary with cost savings estimates
        """
        stats = self._generation_stats
        
        # Estimate token costs (approximation)
        token_cost_per_1k = 0.002  # Approximate cost for GPT-3.5-turbo
        total_cost = (stats["total_api_tokens"] / 1000) * token_cost_per_1k
        
        # Estimate savings from caching
        cache_savings = 0.0
        if self._cache_stats and self._cache_stats["hits"] > 0:
            avg_tokens_per_call = stats["total_api_tokens"] / max(stats["total_api_calls"], 1)
            saved_tokens = self._cache_stats["hits"] * avg_tokens_per_call
            cache_savings = (saved_tokens / 1000) * token_cost_per_1k
        
        # Estimate savings from batch processing
        batch_savings = 0.0
        if stats["batch_operations"] > 0:
            # Rough estimate: batch processing saves ~20% on API calls
            batch_savings = total_cost * 0.2 * (stats["batch_operations"] / max(stats["total_api_calls"], 1))
        
        return {
            "total_estimated_cost": total_cost,
            "cache_savings": cache_savings,
            "batch_savings": batch_savings,
            "total_savings": cache_savings + batch_savings
        }
    
    def is_enabled(self) -> bool:
        """Check if HyQE is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return self.config.ENABLE_HYQE
