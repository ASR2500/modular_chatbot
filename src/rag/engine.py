"""RAG engine for retrieval and generation operations."""

import openai
from typing import List, Dict, Any, Optional
import logging
from src.config import Config
from src.rag.database import DatabaseManager
from src.ner.processor import NERProcessor, Entity
from src.ner_config import NERConfig
from src.query_expansion.processor import QueryExpansionProcessor
from src.query_expansion_config import QueryExpansionConfig
from src.hyde.processor import HyDEProcessor
from src.hyde_config import HyDEConfig
from src.hyqe.processor import HyQEProcessor
from src.hyqe_config import HyQEConfig

logger = logging.getLogger(__name__)

class RAGEngine:
    """Retrieval-Augmented Generation engine."""
    
    def __init__(self, database_manager: DatabaseManager, enable_ner: bool = True, 
                 enable_query_expansion: bool = True, enable_hyde: bool = True, 
                 enable_hyqe: bool = True):
        """Initialize the RAG engine.
        
        Args:
            database_manager: Database manager instance
            enable_ner: Whether to enable Named Entity Recognition
            enable_query_expansion: Whether to enable Query Expansion
            enable_hyde: Whether to enable HyDE (Hypothetical Document Embeddings)
            enable_hyqe: Whether to enable HyQE (Hypothetical Question Embeddings)
        """
        self.db_manager = database_manager
        self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.enable_ner = enable_ner
        self.enable_query_expansion = enable_query_expansion
        self.enable_hyde = enable_hyde
        self.enable_hyqe = enable_hyqe
        self.ner_processor = None
        self.query_expansion_processor = None
        self.hyde_processor = None
        self.hyqe_processor = None
        
        # Initialize NER processor if enabled
        if self.enable_ner:
            try:
                self.ner_processor = NERProcessor()
                logger.info("NER processor initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NER processor: {e}")
                self.enable_ner = False
        
        # Initialize Query Expansion processor if enabled
        if self.enable_query_expansion:
            try:
                self.query_expansion_processor = QueryExpansionProcessor()
                logger.info("Query expansion processor initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Query expansion processor: {e}")
                self.enable_query_expansion = False
        
        # Initialize HyDE processor if enabled
        if self.enable_hyde:
            try:
                self.hyde_processor = HyDEProcessor()
                logger.info("HyDE processor initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize HyDE processor: {e}")
                self.enable_hyde = False
        
        # Initialize HyQE processor if enabled
        if self.enable_hyqe:
            try:
                self.hyqe_processor = HyQEProcessor()
                logger.info("HyQE processor initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize HyQE processor: {e}")
                self.enable_hyqe = False
    
    def retrieve_contexts(self, query: str, n_results: int = None, use_query_expansion: bool = None, 
                         num_expanded_queries: int = None, use_hyde: bool = None, 
                         use_hyqe: bool = None) -> List[Dict[str, Any]]:
        """Retrieve relevant contexts for a query using multiple enhancement techniques.
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            use_query_expansion: Whether to use query expansion
            num_expanded_queries: Number of expanded queries to generate
            use_hyde: Whether to use HyDE for hypothetical document embeddings
            use_hyqe: Whether to use HyQE for hypothetical question embeddings
            
        Returns:
            List of relevant contexts
        """
        n_results = n_results or Config.DEFAULT_N_RESULTS
        use_query_expansion = use_query_expansion if use_query_expansion is not None else self.enable_query_expansion
        use_hyde = use_hyde if use_hyde is not None else self.enable_hyde
        use_hyqe = use_hyqe if use_hyqe is not None else self.enable_hyqe
        num_expanded_queries = num_expanded_queries or Config.QUERY_EXPANSION_NUM_QUERIES
        
        # Step 1: Apply HyDE if enabled - generate hypothetical documents first
        queries_to_process = [query]
        hyde_documents = []
        
        if use_hyde and self.hyde_processor:
            try:
                # Generate hypothetical documents
                hyde_documents = self.hyde_processor.generate_hypothetical_documents(
                    query, domain="python"
                )
                if hyde_documents:
                    # Use hypothetical documents as additional queries
                    queries_to_process.extend(hyde_documents)
                    logger.debug(f"Generated {len(hyde_documents)} hypothetical documents")
            except Exception as e:
                logger.warning(f"HyDE generation failed: {e}")
        
        # Step 2: Apply Query Expansion if enabled
        expanded_queries = []
        if use_query_expansion and self.query_expansion_processor:
            try:
                expanded_queries = self.query_expansion_processor.expand_query(
                    query, num_expansions=num_expanded_queries
                )
                queries_to_process.extend(expanded_queries)
                logger.debug(f"Generated {len(expanded_queries)} expanded queries")
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
        
        # Step 3: Retrieve contexts from both document and question collections
        all_contexts = []
        context_sources = {}  # Track which query generated each context
        
        for i, q in enumerate(queries_to_process):
            try:
                # Determine if we should use both collections or just documents
                if use_hyqe:
                    # Query both document and question collections
                    contexts = self.db_manager.query_both_collections(
                        query=q,
                        n_results=n_results,
                        doc_weight=0.6,  # 60% from documents
                        question_weight=0.4  # 40% from questions
                    )
                else:
                    # Query only document collection
                    contexts = self.db_manager.query_collection(
                        query=q,
                        n_results=n_results
                    )
                
                # Add source information and avoid duplicates
                for ctx in contexts:
                    # Create a unique identifier for the context
                    if ctx.get("source") == "question_embedding":
                        # For question-based results, use the source question as identifier
                        ctx_id = f"q_{ctx.get('metadata', {}).get('source_question', '')[:50]}_{ctx.get('similarity', 0):.3f}"
                    else:
                        # For document-based results, use the question field
                        ctx_id = f"d_{ctx.get('metadata', {}).get('question', '')[:50]}_{ctx.get('similarity', 0):.3f}"
                    
                    if ctx_id not in context_sources:
                        # Determine source type
                        if i == 0:
                            source_type = 'original'
                        elif i <= len(hyde_documents):
                            source_type = 'hyde'
                        else:
                            source_type = 'expanded'
                        
                        # Add retrieval source information
                        if ctx.get("source") == "question_embedding":
                            source_type += "_question"
                        else:
                            source_type += "_document"
                        
                        ctx['query_source'] = source_type
                        ctx['source_query'] = q
                        all_contexts.append(ctx)
                        context_sources[ctx_id] = True
                        
            except Exception as e:
                logger.warning(f"Failed to retrieve contexts for query '{q}': {e}")
        
        # Step 4: Sort by similarity and take top results
        all_contexts.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Filter contexts by relevance threshold
        relevant_contexts = [
            ctx for ctx in all_contexts[:n_results * 2]  # Get more initially
            if ctx.get("similarity", 0) > 0.5  # Minimum similarity threshold
        ]
        
        # Limit to final n_results
        relevant_contexts = relevant_contexts[:n_results]
        
        # Log source distribution
        source_counts = {}
        for ctx in relevant_contexts:
            source = ctx.get('query_source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.debug(f"Retrieved {len(relevant_contexts)} relevant contexts for query: {query[:50]}...")
        logger.debug(f"Source distribution: {source_counts}")
        
        return relevant_contexts
    
    def generate_response(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Generate a response using retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            
        Returns:
            Generated response
        """
        if not contexts:
            return self._generate_no_context_response(query)
        
        # Build context text
        context_text = self._build_context_text(contexts)
        
        # Create the prompt
        prompt = self._create_prompt(query, context_text, contexts)
        
        try:
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": Config.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                presence_penalty=0.1,  # Slight penalty for repetition
                frequency_penalty=0.1  # Slight penalty for frequency
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _build_context_text(self, contexts: List[Dict[str, Any]]) -> str:
        """Build context text from retrieved contexts.
        
        Args:
            contexts: Retrieved contexts
            
        Returns:
            Formatted context text
        """
        context_parts = []
        
        for i, ctx in enumerate(contexts, 1):
            metadata = ctx.get("metadata", {})
            question = metadata.get("question", "")
            answer = metadata.get("answer", "")
            similarity = ctx.get("similarity", 0)
            
            context_part = f"""Context {i} (Relevance: {similarity:.2f}):
Question: {question}
Answer: {answer}"""
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context_text: str, contexts: List[Dict[str, Any]]) -> str:
        """Create the prompt for the LLM.
        
        Args:
            query: User query
            context_text: Formatted context text
            contexts: Retrieved contexts
            
        Returns:
            Formatted prompt
        """
        # Calculate context relevance summary
        avg_similarity = sum(ctx.get("similarity", 0) for ctx in contexts) / len(contexts)
        high_relevance_count = sum(1 for ctx in contexts if ctx.get("similarity", 0) > 0.8)
        
        relevance_note = ""
        if avg_similarity > 0.8:
            relevance_note = "The retrieved contexts are highly relevant to your question."
        elif avg_similarity > 0.6:
            relevance_note = "The retrieved contexts are moderately relevant to your question."
        else:
            relevance_note = "The retrieved contexts have limited relevance to your question."
        
        prompt = f"""Based on the following context from the Python FAQ, please answer the user's question accurately and comprehensively.

{relevance_note}

CONTEXT:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide a direct, accurate answer based on the context provided
2. If the context fully answers the question, provide a comprehensive response
3. If the context partially answers the question, provide what you can and note any limitations
4. If the context doesn't address the question, explain this clearly
5. Use specific examples from the context when helpful
6. Structure your response clearly with proper formatting
7. Reference which context(s) you're using (e.g., "According to Context 1...")
8. If multiple contexts provide different perspectives, acknowledge this
9. Be concise but thorough - aim for clarity over length

Please provide your response:"""

        return prompt
    
    def _generate_no_context_response(self, query: str) -> str:
        """Generate a response when no relevant context is found.
        
        Args:
            query: User query
            
        Returns:
            No-context response
        """
        return f"""I apologize, but I couldn't find relevant information in the Python FAQ database to answer your question about: "{query}"

This could be because:
1. The question is about a topic not covered in the FAQ
2. The question uses terminology that doesn't match the FAQ content
3. The question is too specific or too general for the available content

You might want to:
- Try rephrasing your question with different keywords
- Ask a more specific question about Python fundamentals
- Check the official Python documentation for more comprehensive information

Is there a different Python-related question I can help you with based on the FAQ content?"""
    
    def process_query(self, query: str, n_results: int = None, use_query_expansion: bool = None, 
                     num_expanded_queries: int = None, return_expansion_details: bool = False) -> Dict[str, Any]:
        """Process a complete query with retrieval and generation.
        
        Args:
            query: User query
            n_results: Number of contexts to retrieve
            use_query_expansion: Whether to use query expansion
            num_expanded_queries: Number of expanded queries to generate
            return_expansion_details: Whether to return expansion details
            
        Returns:
            Dictionary with response and contexts
        """
        result = {
            "response": None,
            "contexts": [],
            "query": query,
            "n_contexts": 0
        }
        
        # Add query expansion details if requested
        if return_expansion_details and use_query_expansion and self.query_expansion_processor:
            try:
                expanded_queries = self.query_expansion_processor.expand_query(
                    query, num_expansions=num_expanded_queries or Config.QUERY_EXPANSION_NUM_QUERIES
                )
                result["expanded_queries"] = expanded_queries
                result["expansion_enabled"] = True
            except Exception as e:
                logger.warning(f"Failed to generate expansion details: {e}")
                result["expanded_queries"] = []
                result["expansion_enabled"] = False
        else:
            result["expansion_enabled"] = use_query_expansion and self.query_expansion_processor is not None
        
        # Retrieve contexts
        contexts = self.retrieve_contexts(
            query, 
            n_results=n_results,
            use_query_expansion=use_query_expansion,
            num_expanded_queries=num_expanded_queries
        )
        
        # Generate response
        response = self.generate_response(query, contexts)
        
        result.update({
            "response": response,
            "contexts": contexts,
            "n_contexts": len(contexts)
        })
        
        return result
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG engine.
        
        Returns:
            Dictionary with engine statistics
        """
        db_stats = self.db_manager.get_collection_stats()
        
        stats = {
            "database_stats": db_stats,
            "config": {
                "model": Config.OPENAI_MODEL,
                "embedding_model": Config.EMBEDDING_MODEL,
                "max_tokens": Config.MAX_TOKENS,
                "temperature": Config.TEMPERATURE,
                "default_n_results": Config.DEFAULT_N_RESULTS
            },
            "ner_enabled": self.enable_ner,
            "query_expansion_enabled": self.enable_query_expansion,
            "hyde_enabled": self.enable_hyde
        }
        
        # Add NER stats if available
        if self.enable_ner and self.ner_processor:
            stats["ner_stats"] = self.get_ner_stats()
        
        # Add query expansion stats if available
        if self.enable_query_expansion and self.query_expansion_processor:
            stats["query_expansion_stats"] = {"enabled": True, "processor_available": True}
        
        # Add HyDE stats if available
        if self.enable_hyde and self.hyde_processor:
            stats["hyde_stats"] = self.get_hyde_stats()
        
        return stats
    
    def retrieve_contexts_with_ner(self, query: str, n_results: int = None) -> Dict[str, Any]:
        """Retrieve contexts with NER analysis.
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            
        Returns:
            Dictionary with contexts and NER analysis
        """
        # Standard retrieval
        contexts = self.retrieve_contexts(query, n_results)
        
        # Add NER analysis if enabled
        ner_analysis = {}
        if self.enable_ner and self.ner_processor:
            try:
                ner_analysis = self.ner_processor.extract_entities_from_query_and_context(query, contexts)
            except Exception as e:
                logger.warning(f"NER analysis failed: {e}")
                ner_analysis = {"error": str(e)}
        
        return {
            "contexts": contexts,
            "ner_analysis": ner_analysis,
            "query": query,
            "n_contexts": len(contexts)
        }
    
    def generate_response_with_ner(self, query: str, contexts: List[Dict[str, Any]], 
                                  ner_analysis: Dict[str, Any] = None) -> str:
        """Generate a response with NER-enhanced context.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            ner_analysis: NER analysis results
            
        Returns:
            Generated response with NER insights
        """
        if not contexts:
            return self._generate_no_context_response(query)
        
        # Build enhanced context with NER insights
        context_text = self._build_context_text_with_ner(contexts, ner_analysis)
        
        # Create enhanced prompt
        prompt = self._create_prompt_with_ner(query, context_text, contexts, ner_analysis)
        
        try:
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": Config.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating NER-enhanced response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _build_context_text_with_ner(self, contexts: List[Dict[str, Any]], 
                                    ner_analysis: Dict[str, Any] = None) -> str:
        """Build context text with NER insights.
        
        Args:
            contexts: Retrieved contexts
            ner_analysis: NER analysis results
            
        Returns:
            Enhanced context text
        """
        context_parts = []
        
        for i, ctx in enumerate(contexts, 1):
            metadata = ctx.get("metadata", {})
            question = metadata.get("question", "")
            answer = metadata.get("answer", "")
            similarity = ctx.get("similarity", 0)
            
            # Add NER insights for this context
            ner_info = ""
            if ner_analysis and "contexts" in ner_analysis:
                try:
                    context_entities = ner_analysis["contexts"][i-1].get("entities", [])
                    if context_entities:
                        entities_text = ", ".join([f"{e.text} ({e.label})" for e in context_entities[:5]])
                        ner_info = f"\nKey entities: {entities_text}"
                except (IndexError, KeyError, AttributeError):
                    pass
            
            context_part = f"""Context {i} (Relevance: {similarity:.2f}):
Question: {question}
Answer: {answer}{ner_info}"""
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _create_prompt_with_ner(self, query: str, context_text: str, 
                               contexts: List[Dict[str, Any]], ner_analysis: Dict[str, Any] = None) -> str:
        """Create an enhanced prompt with NER insights.
        
        Args:
            query: User query
            context_text: Formatted context text
            contexts: Retrieved contexts
            ner_analysis: NER analysis results
            
        Returns:
            Enhanced prompt
        """
        # Build NER insights
        ner_insights = ""
        if ner_analysis and "query" in ner_analysis:
            query_entities = ner_analysis["query"]
            if query_entities:
                entities_text = ", ".join([f"{e.text} ({e.label})" for e in query_entities])
                ner_insights = f"\nKey entities in the question: {entities_text}"
        
        # Calculate context relevance
        avg_similarity = sum(ctx.get("similarity", 0) for ctx in contexts) / len(contexts)
        
        relevance_note = ""
        if avg_similarity > 0.8:
            relevance_note = "The retrieved contexts are highly relevant to your question."
        elif avg_similarity > 0.6:
            relevance_note = "The retrieved contexts are moderately relevant to your question."
        else:
            relevance_note = "The retrieved contexts have limited relevance to your question."
        
        prompt = f"""Based on the following context from the Python FAQ, please answer the user's question accurately and comprehensively.

{relevance_note}{ner_insights}

CONTEXT:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide a direct, accurate answer based on the context provided
2. Pay special attention to the key entities mentioned in the question
3. If the context fully answers the question, provide a comprehensive response
4. Use specific examples from the context when helpful
5. Structure your response clearly with proper formatting
6. Reference which context(s) you're using (e.g., "According to Context 1...")
7. If multiple contexts provide different perspectives, acknowledge this
8. Be concise but thorough - focus on the most relevant information

Please provide your response:"""

        return prompt
    
    def process_query_with_ner(self, query: str, n_results: int = None, 
                              use_query_expansion: bool = None, num_expanded_queries: int = None,
                              return_expansion_details: bool = False) -> Dict[str, Any]:
        """Process a complete query with NER-enhanced retrieval and generation.
        
        Args:
            query: User query
            n_results: Number of contexts to retrieve
            use_query_expansion: Whether to use query expansion
            num_expanded_queries: Number of expanded queries to generate
            return_expansion_details: Whether to return expansion details
            
        Returns:
            Dictionary with response, contexts, and NER analysis
        """
        # Retrieve contexts with NER
        retrieval_result = self.retrieve_contexts_with_ner(query, n_results)
        contexts = retrieval_result["contexts"]
        ner_analysis = retrieval_result["ner_analysis"]
        
        # If query expansion is enabled, get expanded queries and merge contexts
        expanded_queries = []
        if use_query_expansion and self.query_expansion_processor:
            try:
                expanded_queries = self.query_expansion_processor.expand_query(
                    query, num_expansions=num_expanded_queries or Config.QUERY_EXPANSION_NUM_QUERIES
                )
                
                # Get additional contexts from expanded queries
                for expanded_query in expanded_queries:
                    try:
                        expanded_contexts = self.retrieve_contexts(
                            expanded_query, n_results=n_results
                        )
                        # Add source information and merge unique contexts
                        for ctx in expanded_contexts:
                            ctx_id = f"{ctx.get('metadata', {}).get('question', '')[:50]}_{ctx.get('similarity', 0):.3f}"
                            # Check if this context is already in our results
                            if not any(existing_ctx_id == ctx_id for existing_ctx_id in 
                                     [f"{existing_ctx.get('metadata', {}).get('question', '')[:50]}_{existing_ctx.get('similarity', 0):.3f}" 
                                      for existing_ctx in contexts]):
                                ctx['query_source'] = 'expanded'
                                ctx['source_query'] = expanded_query
                                contexts.append(ctx)
                    except Exception as e:
                        logger.warning(f"Failed to retrieve contexts for expanded query '{expanded_query}': {e}")
                
                # Re-sort and limit contexts
                contexts.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                contexts = contexts[:n_results or Config.DEFAULT_N_RESULTS]
                
            except Exception as e:
                logger.warning(f"Query expansion failed in NER processing: {e}")
        
        # Generate enhanced response
        response = self.generate_response_with_ner(query, contexts, ner_analysis)
        
        result = {
            "response": response,
            "contexts": contexts,
            "ner_analysis": ner_analysis,
            "query": query,
            "n_contexts": len(contexts),
            "ner_enabled": self.enable_ner,
            "expansion_enabled": use_query_expansion and self.query_expansion_processor is not None
        }
        
        # Add expansion details if requested
        if return_expansion_details:
            result["expanded_queries"] = expanded_queries
        
        return result
    
    def get_ner_stats(self) -> Dict[str, Any]:
        """Get NER processor statistics.
        
        Returns:
            Dictionary with NER statistics
        """
        if not self.enable_ner or not self.ner_processor:
            return {"enabled": False, "error": "NER processor not available"}
        
        try:
            return {
                "enabled": True,
                "processor_stats": self.ner_processor.get_processor_stats()
            }
        except Exception as e:
            return {"enabled": False, "error": str(e)}
    
    def get_hyde_stats(self) -> Dict[str, Any]:
        """Get HyDE processor statistics.
        
        Returns:
            Dictionary with HyDE statistics
        """
        if not self.enable_hyde or not self.hyde_processor:
            return {"enabled": False, "error": "HyDE processor not available"}
        
        try:
            return {
                "enabled": True,
                "processor_stats": self.hyde_processor.get_stats()
            }
        except Exception as e:
            return {"enabled": False, "error": str(e)}

    def expand_query(self, query: str) -> str:
        """Expand the query using query expansion techniques.
        
        Args:
            query: User query
            
        Returns:
            Expanded query
        """
        if not self.enable_query_expansion or not self.query_expansion_processor:
            return query  # No expansion performed
        
        try:
            # Perform query expansion
            expanded_query = self.query_expansion_processor.expand_query(query)
            
            logger.info(f"Query expanded from '{query}' to '{expanded_query}'")
            
            return expanded_query
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query  # Return original query on failure
    
    def generate_and_populate_question_embeddings(self, documents: List[Dict[str, Any]], 
                                                 progress_callback=None) -> bool:
        """Generate hypothetical questions for documents and populate the questions collection.
        
        Args:
            documents: List of document dictionaries
            progress_callback: Optional callback function for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_hyqe or not self.hyqe_processor:
            logger.warning("HyQE is not enabled or processor not available")
            return False
        
        try:
            logger.info(f"Generating hypothetical questions for {len(documents)} documents")
            
            # Load existing content hashes to avoid regenerating unchanged content
            self.hyqe_processor.load_content_hashes()
            
            # Prepare chunks for question generation
            chunks = []
            for i, doc in enumerate(documents):
                chunk = {
                    "content": doc.get("text", ""),
                    "id": doc.get("id", f"doc_{i}"),
                    "metadata": doc.get("metadata", {})
                }
                chunks.append(chunk)
            
            # Generate questions for all chunks with optimization
            all_questions = self.hyqe_processor.generate_questions_for_chunks(
                chunks, 
                domain="python",
                progress_callback=progress_callback
            )
            
            if not all_questions:
                logger.warning("No questions were generated")
                return False
            
            # Populate the questions collection with optimization
            if hasattr(self.db_manager, 'populate_questions_collection_optimized'):
                final_count = self.db_manager.populate_questions_collection_optimized(
                    questions=all_questions,
                    progress_callback=progress_callback
                )
            else:
                # Fall back to regular method
                self.db_manager.populate_questions_collection(
                    questions=all_questions,
                    progress_callback=progress_callback
                )
                # Get final count manually
                questions_collection = self.db_manager.get_or_create_questions_collection()
                final_count = questions_collection.count()
            
            # Save content hashes for future runs
            self.hyqe_processor.save_content_hashes()
            
            logger.info(f"Successfully generated and stored questions. Collection now has {final_count} total questions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate and populate question embeddings: {str(e)}")
            return False
    
    def get_hyqe_stats(self) -> Dict[str, Any]:
        """Get HyQE processor statistics.
        
        Returns:
            Dictionary with HyQE statistics
        """
        if not self.enable_hyqe or not self.hyqe_processor:
            return {"enabled": False, "error": "HyQE processor not available"}
        
        try:
            stats = self.hyqe_processor.get_stats()
            
            # Add questions collection stats
            questions_stats = self.db_manager.get_questions_collection_stats()
            stats["questions_collection"] = questions_stats
            
            return {
                "enabled": True,
                "processor_stats": stats
            }
        except Exception as e:
            return {"enabled": False, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics.
        
        Returns:
            Dictionary with all engine statistics
        """
        return {
            "database_stats": self.db_manager.get_stats(),
            "config": {
                "model": Config.OPENAI_MODEL,
                "embedding_model": Config.EMBEDDING_MODEL,
                "default_n_results": Config.DEFAULT_N_RESULTS,
                "temperature": Config.TEMPERATURE,
                "max_tokens": Config.MAX_TOKENS
            },
            "ner": self.get_ner_stats(),
            "query_expansion": {
                "enabled": self.enable_query_expansion,
                "processor_available": self.query_expansion_processor is not None
            },
            "hyde": self.get_hyde_stats(),
            "hyqe": self.get_hyqe_stats()
        }
