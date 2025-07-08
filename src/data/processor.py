"""Data processing module for loading and preparing FAQ data."""

import pandas as pd
import polars as pl
from typing import List, Dict, Any
import logging
from src.config import Config

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading and processing operations."""
    
    def __init__(self, data_path: str = None):
        """Initialize the data processor.
        
        Args:
            data_path: Path to the CSV file containing FAQ data
        """
        self.data_path = data_path or Config.DATA_PATH
        self._cached_data = None
    
    def load_and_process_data(self) -> pl.DataFrame:
        """Load and process the Python FAQ data.
        
        Returns:
            Processed DataFrame with FAQ data
        """
        if self._cached_data is not None:
            return self._cached_data
        
        try:
            # Load data with proper encoding handling
            df_pandas = self._load_csv_with_encoding()
            faq_data = pl.from_pandas(df_pandas)
            
            # Clean and prepare data
            faq_data = self._clean_data(faq_data)
            
            # Cache the processed data
            self._cached_data = faq_data
            
            logger.info(f"Successfully loaded {len(faq_data)} FAQ entries")
            return faq_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _load_csv_with_encoding(self) -> pd.DataFrame:
        """Load CSV with proper encoding handling.
        
        Returns:
            Pandas DataFrame with loaded data
        """
        encodings = ["utf-8", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                return pd.read_csv(self.data_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        # Final fallback with error handling
        logger.warning("Using fallback encoding with error handling")
        return pd.read_csv(self.data_path, encoding="utf-8", errors="ignore")
    
    def _clean_data(self, faq_data: pl.DataFrame) -> pl.DataFrame:
        """Clean and prepare the FAQ data.
        
        Args:
            faq_data: Raw FAQ data
            
        Returns:
            Cleaned DataFrame
        """
        # Clean and prepare data
        faq_data = faq_data.with_columns([
            pl.col("Questions").str.strip_chars().str.replace_all(r"¶", "").alias("question"),
            pl.col("Answers").str.strip_chars().alias("answer")
        ])
        
        # Filter out empty rows
        faq_data = faq_data.filter(
            (pl.col("question").is_not_null()) & 
            (pl.col("answer").is_not_null()) &
            (pl.col("question") != "") &
            (pl.col("answer") != "") &
            (pl.col("question").str.len_chars() > 5) &  # Minimum question length
            (pl.col("answer").str.len_chars() > 10)     # Minimum answer length
        )
        
        # Remove duplicate questions
        faq_data = faq_data.unique(subset=["question"])
        
        return faq_data
    
    def prepare_documents_for_embedding(self, faq_data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Prepare documents for embedding.
        
        Args:
            faq_data: Processed FAQ data
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        for i, row in enumerate(faq_data.iter_rows(named=True)):
            # Create enhanced document text for better retrieval
            combined_text = f"Question: {row['question']}\n\nAnswer: {row['answer']}"
            
            # Add keywords from question for better matching
            keywords = self._extract_keywords(row['question'])
            if keywords:
                combined_text += f"\n\nKeywords: {', '.join(keywords)}"
            
            # Create metadata (ensuring ChromaDB compatibility)
            metadata = {
                "question": row['question'],
                "answer": row['answer'],
                "index": i,
                "keywords": ', '.join(keywords) if keywords else ""  # Convert list to string
            }
            
            documents.append({
                "id": f"faq_{i}",
                "text": combined_text,
                "metadata": metadata
            })
        
        return documents
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from a question.
        
        Args:
            question: Question text
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - can be enhanced with NLP
        import re
        
        # Remove common stop words and extract meaningful terms
        stop_words = {"what", "is", "the", "how", "why", "when", "where", "which", "who", "does", "do", "can", "will", "should", "would", "could", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        # Extract words (alphanumeric, allowing underscores for Python terms)
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', question.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to 10 keywords
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        if self._cached_data is None:
            self.load_and_process_data()
        
        data = self._cached_data
        
        return {
            "total_entries": len(data),
            "avg_question_length": data["question"].str.len_chars().mean(),
            "avg_answer_length": data["answer"].str.len_chars().mean(),
            "longest_question": data["question"].str.len_chars().max(),
            "longest_answer": data["answer"].str.len_chars().max(),
        }
