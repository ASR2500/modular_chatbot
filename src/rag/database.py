"""Database manager for persistent ChromaDB operations."""

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from typing import List, Dict, Any, Optional
import logging
import os
import time
from src.config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages ChromaDB operations with persistence."""
    
    def __init__(self, persist_dir: str = None):
        """Initialize the database manager.
        
        Args:
            persist_dir: Directory for persistent storage
        """
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        self.collection_name = Config.COLLECTION_NAME
        self._client = None
        self._collection = None
        self._embedding_function = None
    
    @property
    def client(self):
        """Get or create the ChromaDB client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    @property
    def embedding_function(self):
        """Get or create the embedding function."""
        if self._embedding_function is None:
            self._embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=Config.OPENAI_API_KEY,
                model_name=Config.EMBEDDING_MODEL
            )
        return self._embedding_function
    
    def _create_client(self):
        """Create a ChromaDB client with persistence."""
        try:
            # Create persistence directory if it doesn't exist
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Create persistent client
            client = chromadb.PersistentClient(path=self.persist_dir)
            
            logger.info(f"Created persistent ChromaDB client at {self.persist_dir}")
            return client
            
        except Exception as e:
            logger.error(f"Error creating ChromaDB client: {str(e)}")
            # Fallback to in-memory client
            logger.warning("Falling back to in-memory client")
            return chromadb.Client()
    
    def get_or_create_collection(self, force_recreate: bool = False):
        """Get or create the FAQ collection.
        
        Args:
            force_recreate: Whether to force recreation of the collection
            
        Returns:
            ChromaDB collection
        """
        if self._collection is not None and not force_recreate:
            return self._collection
        
        try:
            if force_recreate:
                self._delete_collection()
            
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            logger.info(f"Retrieved existing collection '{self.collection_name}' with {collection.count()} documents")
            
        except Exception as e:
            # Collection doesn't exist or has issues, create new one
            logger.info(f"Creating new collection '{self.collection_name}'")
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        
        self._collection = collection
        return collection
    
    def _delete_collection(self):
        """Delete the existing collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.warning(f"Could not delete collection: {str(e)}")
    
    def populate_collection(self, documents: List[Dict[str, Any]], 
                          batch_size: int = 10, 
                          progress_callback=None):
        """Populate the collection with documents.
        
        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process in each batch
            progress_callback: Optional callback function for progress updates
        """
        collection = self.get_or_create_collection()
        
        # Check if collection already has documents
        if collection.count() > 0:
            logger.info(f"Collection already contains {collection.count()} documents")
            return
        
        logger.info(f"Populating collection with {len(documents)} documents")
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare batch data
            batch_docs = [doc["text"] for doc in batch]
            batch_ids = [doc["id"] for doc in batch]
            batch_metadata = [doc["metadata"] for doc in batch]
            
            # Add to collection
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metadata
            )
            
            # Report progress
            processed = min(i + batch_size, len(documents))
            progress = processed / len(documents)
            
            if progress_callback:
                progress_callback(progress)
            
            logger.debug(f"Processed {processed}/{len(documents)} documents")
        
        logger.info(f"Successfully populated collection with {collection.count()} documents")
    
    def query_collection(self, query: str, n_results: int = 3, 
                        filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Query the collection for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant documents with metadata
        """
        collection = self.get_or_create_collection()
        
        if collection.count() == 0:
            logger.warning("Collection is empty, cannot perform query")
            return []
        
        try:
            # Perform the query
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count()),
                include=["documents", "metadatas", "distances"],
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                formatted_results.append({
                    "document": doc,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": i + 1,
                    "similarity": 1 - distance  # Convert distance to similarity
                })
            
            logger.debug(f"Found {len(formatted_results)} relevant documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.get_or_create_collection()
            count = collection.count()
            
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "persist_dir": self.persist_dir,
                "is_persistent": self.persist_dir != ":memory:"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        try:
            self._delete_collection()
            self._collection = None
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self._client:
            # ChromaDB doesn't have an explicit close method
            # but we can clear our references
            self._client = None
            self._collection = None
            logger.info("Database connection closed")
    
    def get_or_create_questions_collection(self, questions_collection_name: str = None):
        """Get or create the questions collection.
        
        Args:
            questions_collection_name: Name for the questions collection
            
        Returns:
            ChromaDB collection for questions
        """
        if questions_collection_name is None:
            questions_collection_name = self.collection_name + "_questions"
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=questions_collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Retrieved existing questions collection '{questions_collection_name}' with {collection.count()} questions")
            return collection
        
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=questions_collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Created new questions collection '{questions_collection_name}'")
            return collection
    
    def populate_questions_collection(self, questions: List[Dict[str, Any]], 
                                    questions_collection_name: str = None,
                                    batch_size: int = 10, 
                                    progress_callback=None):
        """Populate the questions collection with generated questions.
        
        Args:
            questions: List of question dictionaries
            questions_collection_name: Name for the questions collection
            batch_size: Number of questions to process in each batch
            progress_callback: Optional callback function for progress updates
        """
        collection = self.get_or_create_questions_collection(questions_collection_name)
        
        # Check if collection already has questions
        if collection.count() > 0:
            logger.info(f"Questions collection already contains {collection.count()} questions")
            return
        
        logger.info(f"Populating questions collection with {len(questions)} questions")
        
        # Process questions in batches
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            
            # Prepare batch data
            batch_questions = [q["question"] for q in batch]
            batch_ids = [f"q_{q.get('chunk_id', 'unknown')}_{q.get('question_index', i)}" for q in batch]
            batch_metadata = [
                {
                    "question": q["question"],
                    "chunk_id": q.get("chunk_id", "unknown"),
                    "chunk_content": q.get("chunk_content", ""),
                    "question_index": q.get("question_index", 0),
                    "generation_method": q.get("generation_method", "hyqe"),
                    "question_style": q.get("config", {}).get("style", "unknown"),
                    "question_quality": q.get("config", {}).get("quality", "unknown"),
                    "type": "question"  # Mark as question for identification
                }
                for q in batch
            ]
            
            # Validate metadata for ChromaDB compatibility
            batch_metadata = [Config.validate_metadata(meta) for meta in batch_metadata]
            
            # Add to collection
            collection.add(
                documents=batch_questions,
                ids=batch_ids,
                metadatas=batch_metadata
            )
            
            # Report progress
            processed = min(i + batch_size, len(questions))
            progress = processed / len(questions)
            
            if progress_callback:
                progress_callback(progress)
            
            logger.debug(f"Processed {processed}/{len(questions)} questions")
        
        logger.info(f"Successfully populated questions collection with {collection.count()} questions")
    
    def query_questions_collection(self, query: str, n_results: int = 5, 
                                 questions_collection_name: str = None) -> List[Dict[str, Any]]:
        """Query the questions collection.
        
        Args:
            query: Query string
            n_results: Number of results to return
            questions_collection_name: Name of the questions collection
            
        Returns:
            List of question results with metadata
        """
        collection = self.get_or_create_questions_collection(questions_collection_name)
        
        if collection.count() == 0:
            logger.warning("Questions collection is empty")
            return []
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    
                    result = {
                        "question": doc,
                        "metadata": metadata,
                        "similarity": 1 - results["distances"][0][i] if results["distances"] else 0.0,
                        "source": "question_embedding"
                    }
                    formatted_results.append(result)
            
            logger.debug(f"Found {len(formatted_results)} question matches for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying questions collection: {str(e)}")
            return []
    
    def query_both_collections(self, query: str, n_results: int = 5, 
                             questions_collection_name: str = None,
                             doc_weight: float = 0.6, question_weight: float = 0.4) -> List[Dict[str, Any]]:
        """Query both document and questions collections and merge results.
        
        Args:
            query: Query string
            n_results: Total number of results to return
            questions_collection_name: Name of the questions collection
            doc_weight: Weight for document results (0.0-1.0)
            question_weight: Weight for question results (0.0-1.0)
            
        Returns:
            List of merged results from both collections
        """
        # Calculate how many results to get from each collection
        doc_results_count = max(1, int(n_results * doc_weight))
        question_results_count = max(1, int(n_results * question_weight))
        
        # Query both collections
        doc_results = self.query_collection(query, doc_results_count)
        question_results = self.query_questions_collection(query, question_results_count, questions_collection_name)
        
        # Merge and sort results
        all_results = []
        
        # Add document results
        for result in doc_results:
            result["source"] = "document_embedding"
            all_results.append(result)
        
        # Add question results, but get the original document content
        for result in question_results:
            # For question results, we want to return the original document content
            # not the question itself
            metadata = result.get("metadata", {})
            
            # Handle both old and new metadata structures
            chunk_content = (
                metadata.get("chunk_content") or  # New structure
                metadata.get("source_content_preview") or  # Old structure
                ""
            )
            
            if chunk_content:
                modified_result = {
                    "document": chunk_content,
                    "metadata": {
                        "question": result["question"],
                        "answer": chunk_content,
                        "source_question": result["question"],
                        "chunk_content": chunk_content,  # Add chunk_content to metadata
                        "chunk_id": metadata.get("chunk_id") or metadata.get("source_chunk_id", "unknown"),
                        "generation_method": metadata.get("generation_method", "hyqe"),
                        "question_style": metadata.get("question_style", "unknown")
                    },
                    "similarity": result["similarity"],
                    "source": "question_embedding"
                }
                all_results.append(modified_result)
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Return top n_results
        return all_results[:n_results]
    
    def delete_questions_collection(self, questions_collection_name: str = None):
        """Delete the questions collection.
        
        Args:
            questions_collection_name: Name of the questions collection to delete
        """
        if questions_collection_name is None:
            questions_collection_name = self.collection_name + "_questions"
        
        try:
            self.client.delete_collection(name=questions_collection_name)
            logger.info(f"Deleted questions collection '{questions_collection_name}'")
        except Exception as e:
            logger.warning(f"Could not delete questions collection: {str(e)}")
    
    def get_questions_collection_stats(self, questions_collection_name: str = None) -> Dict[str, Any]:
        """Get statistics about the questions collection.
        
        Args:
            questions_collection_name: Name of the questions collection
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.get_or_create_questions_collection(questions_collection_name)
            
            return {
                "name": questions_collection_name or (self.collection_name + "_questions"),
                "question_count": collection.count(),
                "is_persistent": os.path.exists(self.persist_dir)
            }
        except Exception as e:
            logger.error(f"Error getting questions collection stats: {str(e)}")
            return {
                "name": questions_collection_name or (self.collection_name + "_questions"),
                "question_count": 0,
                "is_persistent": False,
                "error": str(e)
            }
    
    def populate_questions_collection_optimized(self, questions: List[Dict[str, Any]], 
                                              questions_collection_name: str = None,
                                              batch_size: int = 20, 
                                              progress_callback=None,
                                              skip_existing: bool = True):
        """Optimized method to populate questions collection with deduplication.
        
        Args:
            questions: List of question dictionaries
            questions_collection_name: Name for the questions collection
            batch_size: Number of questions to process in each batch
            progress_callback: Optional callback function for progress updates
            skip_existing: Whether to skip if collection already has questions
        """
        collection = self.get_or_create_questions_collection(questions_collection_name)
        
        # Check if collection already has questions and skip if requested
        existing_count = collection.count()
        if skip_existing and existing_count > 0:
            logger.info(f"Questions collection already contains {existing_count} questions, skipping population")
            return existing_count
        
        logger.info(f"Populating questions collection with {len(questions)} questions (existing: {existing_count})")
        
        # Get existing question IDs to avoid duplicates
        existing_ids = set()
        if existing_count > 0:
            try:
                existing_data = collection.get()
                existing_ids = set(existing_data.get("ids", []))
            except Exception as e:
                logger.warning(f"Failed to get existing IDs: {str(e)}")
        
        # Filter out questions that already exist
        new_questions = []
        for q in questions:
            question_id = f"q_{q.get('chunk_id', 'unknown')}_{q.get('question_index', 0)}"
            if question_id not in existing_ids:
                new_questions.append(q)
        
        if not new_questions:
            logger.info("No new questions to add")
            return existing_count
        
        logger.info(f"Adding {len(new_questions)} new questions (filtered {len(questions) - len(new_questions)} duplicates)")
        
        # Process questions in batches
        added_count = 0
        for i in range(0, len(new_questions), batch_size):
            batch = new_questions[i:i + batch_size]
            
            try:
                # Prepare batch data
                batch_questions = [q["question"] for q in batch]
                batch_ids = [f"q_{q.get('chunk_id', 'unknown')}_{q.get('question_index', i)}" for q in batch]
                batch_metadata = [
                    {
                        "question": q["question"],
                        "chunk_id": q.get("chunk_id", "unknown"),
                        "chunk_content": q.get("chunk_content", "")[:500],  # Limit chunk content size
                        "question_index": q.get("question_index", 0),
                        "generation_method": q.get("generation_method", "hyqe"),
                        "question_style": q.get("config", {}).get("style", "unknown"),
                        "question_quality": q.get("config", {}).get("quality", "unknown"),
                        "type": "question",  # Mark as question for identification
                        "created_at": str(time.time())  # Add timestamp
                    }
                    for q in batch
                ]
                
                # Validate metadata for ChromaDB compatibility
                batch_metadata = [Config.validate_metadata(meta) for meta in batch_metadata]
                
                # Add to collection
                collection.add(
                    documents=batch_questions,
                    ids=batch_ids,
                    metadatas=batch_metadata
                )
                
                added_count += len(batch)
                
                # Update progress
                if progress_callback:
                    progress = (i + len(batch)) / len(new_questions)
                    progress_callback(progress)
                
                logger.debug(f"Added batch {i//batch_size + 1}/{(len(new_questions) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Failed to add batch {i//batch_size + 1}: {str(e)}")
                continue
        
        total_count = existing_count + added_count
        logger.info(f"Questions collection now contains {total_count} questions ({added_count} new)")
        
        return total_count
