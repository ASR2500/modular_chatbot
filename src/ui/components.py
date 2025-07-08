"""Streamlit UI components for the Python FAQ chatbot."""

import streamlit as st
from typing import List, Dict, Any, Optional
import logging
from src.config import Config

logger = logging.getLogger(__name__)

class ChatbotUI:
    """Streamlit UI for the chatbot."""
    
    def __init__(self):
        """Initialize the UI components."""
        self.setup_page_config()
    
    def setup_page_config(self):
        """Set up the Streamlit page configuration."""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout="wide"
        )
    
    def render_header(self):
        """Render the main header."""
        st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
        st.markdown(
            "Ask me anything about Python! I'll search through the Python FAQ database "
            "to provide accurate and helpful answers."
        )
    
    def render_sidebar(self, data_stats: Dict[str, Any], 
                      engine_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Render the sidebar with information and controls.
        
        Args:
            data_stats: Statistics about the data
            engine_stats: Statistics about the RAG engine
            
        Returns:
            Dictionary with user settings
        """
        with st.sidebar:
            st.header("ℹ️ Information")
            
            # Data information
            st.subheader("📊 Data Statistics")
            if data_stats and 'total_entries' in data_stats:
                st.markdown(f"**Total FAQ entries**: {data_stats.get('total_entries', 'N/A')}")
            else:
                st.markdown("**Total FAQ entries**: Loading...")
            
            if engine_stats and 'database_stats' in engine_stats:
                db_stats = engine_stats['database_stats']
                st.markdown(f"**Vector database size**: {db_stats.get('document_count', 'N/A')}")
                st.markdown(f"**Persistence**: {'Yes' if db_stats.get('is_persistent', False) else 'No'}")
            else:
                st.markdown("**Vector database size**: Loading...")
                st.markdown("**Persistence**: Loading...")
            
            # Configuration
            st.subheader("🔧 Settings")
            if engine_stats and 'config' in engine_stats:
                config_stats = engine_stats['config']
                st.markdown(f"**Model**: {config_stats.get('model', 'N/A')}")
                st.markdown(f"**Embedding**: {config_stats.get('embedding_model', 'N/A')}")
            else:
                st.markdown(f"**Model**: {Config.OPENAI_MODEL}")
                st.markdown(f"**Embedding**: {Config.EMBEDDING_MODEL}")
            
            # User controls
            st.subheader("🎛️ Controls")
            n_results = st.slider(
                "Number of contexts to retrieve", 
                min_value=1, 
                max_value=10, 
                value=Config.DEFAULT_N_RESULTS,
                help="More contexts provide more information but may include less relevant content"
            )
            
            temperature = st.slider(
                "Response creativity",
                min_value=0.0,
                max_value=1.0,
                value=Config.TEMPERATURE,
                step=0.1,
                help="Lower values make responses more focused and deterministic"
            )
            
            # NER controls
            st.subheader("🧠 Named Entity Recognition")
            enable_ner = st.checkbox(
                "Enable NER Analysis", 
                value=True,
                help="Use Named Entity Recognition to identify key entities in queries and responses"
            )
            
            show_ner_details = st.checkbox(
                "Show NER Details",
                value=False,
                help="Display detailed NER analysis in responses"
            )
            
            # Query Expansion controls
            st.subheader("🔍 Query Expansion")
            enable_query_expansion = st.checkbox(
                "Enable Query Expansion",
                value=True,
                help="Generate alternative queries to improve search results"
            )
            
            num_expanded_queries = st.slider(
                "Number of expanded queries",
                min_value=1,
                max_value=10,
                value=3,
                help="How many alternative queries to generate"
            )
            
            show_expansion_details = st.checkbox(
                "Show Expansion Details",
                value=False,
                help="Display the generated expanded queries"
            )
            
            # HyDE controls
            st.subheader("📝 HyDE (Hypothetical Document Embeddings)")
            enable_hyde = st.checkbox(
                "Enable HyDE",
                value=True,
                help="Generate hypothetical documents to improve retrieval"
            )
            
            if enable_hyde:
                col1, col2 = st.columns(2)
                with col1:
                    hyde_temperature = st.slider(
                        "HyDE Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        step=0.1,
                        help="Controls creativity of hypothetical documents (0.2-0.3 for factual, 0.7+ for creative)"
                    )
                    
                    hyde_max_tokens = st.slider(
                        "HyDE Max Tokens",
                        min_value=50,
                        max_value=500,
                        value=150,
                        step=25,
                        help="Maximum length of hypothetical documents"
                    )
                
                with col2:
                    hyde_answer_style = st.selectbox(
                        "Answer Style",
                        options=["concise", "detailed", "steps", "bullets"],
                        index=0,
                        help="Style of hypothetical documents"
                    )
                    
                    hyde_include_examples = st.checkbox(
                        "Include Examples",
                        value=True,
                        help="Include code examples in hypothetical documents"
                    )
                
                hyde_batch_mode = st.checkbox(
                    "Batch Mode",
                    value=False,
                    help="Generate multiple hypothetical documents for ensemble embedding"
                )
                
                if hyde_batch_mode:
                    hyde_batch_size = st.slider(
                        "Batch Size",
                        min_value=2,
                        max_value=5,
                        value=3,
                        help="Number of hypothetical documents to generate"
                    )
                else:
                    hyde_batch_size = 1
                
                show_hyde_details = st.checkbox(
                    "Show HyDE Details",
                    value=False,
                    help="Display generated hypothetical documents"
                )
            else:
                # Default values when HyDE is disabled
                hyde_temperature = 0.2
                hyde_max_tokens = 150
                hyde_answer_style = "concise"
                hyde_include_examples = True
                hyde_batch_mode = False
                hyde_batch_size = 1
                show_hyde_details = False
            
            # HyQE controls
            st.subheader("❓ HyQE (Hypothetical Question Embeddings)")
            enable_hyqe = st.checkbox(
                "Enable HyQE",
                value=True,
                help="Use hypothetical questions to improve retrieval"
            )
            
            if enable_hyqe:
                col1, col2 = st.columns(2)
                with col1:
                    hyqe_questions_per_chunk = st.slider(
                        "Questions per Chunk",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="Number of questions to generate per document chunk"
                    )
                    
                    hyqe_temperature = st.slider(
                        "HyQE Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.4,
                        step=0.1,
                        help="Controls creativity in question generation"
                    )
                
                with col2:
                    hyqe_question_style = st.selectbox(
                        "Question Style",
                        options=["natural", "instructional", "faq", "exam-style"],
                        index=0,
                        help="Style of questions to generate"
                    )
                    
                    hyqe_quality = st.selectbox(
                        "Question Quality",
                        options=["vague", "balanced", "specific"],
                        index=1,
                        help="Specificity level of generated questions"
                    )
                
                hyqe_include_chunk_summary = st.checkbox(
                    "Include Chunk Summary",
                    value=True,
                    help="Include content summary when generating questions"
                )
                
                show_hyqe_details = st.checkbox(
                    "Show HyQE Details",
                    value=False,
                    help="Display retrieval source information (questions vs documents)"
                )
            else:
                # Default values when HyQE is disabled
                hyqe_questions_per_chunk = 3
                hyqe_temperature = 0.4
                hyqe_question_style = "natural"
                hyqe_quality = "balanced"
                hyqe_include_chunk_summary = True
                show_hyqe_details = False
            
            # Action buttons
            st.subheader("🔄 Actions")
            clear_chat = st.button("Clear Chat History", help="Clear all chat messages")
            
            reset_db = st.button(
                "Reset Vector Database", 
                help="Delete and recreate the vector database"
            )
            
            # Advanced information
            with st.expander("🔍 Advanced Information"):
                st.json(data_stats)
                st.json(engine_stats)
            
            return {
                "n_results": n_results,
                "temperature": temperature,
                "enable_ner": enable_ner,
                "show_ner_details": show_ner_details,
                "enable_query_expansion": enable_query_expansion,
                "num_expanded_queries": num_expanded_queries,
                "show_expansion_details": show_expansion_details,
                "enable_hyde": enable_hyde,
                "hyde_temperature": hyde_temperature,
                "hyde_max_tokens": hyde_max_tokens,
                "hyde_answer_style": hyde_answer_style,
                "hyde_include_examples": hyde_include_examples,
                "hyde_batch_mode": hyde_batch_mode,
                "hyde_batch_size": hyde_batch_size,
                "show_hyde_details": show_hyde_details,
                "enable_hyqe": enable_hyqe,
                "hyqe_questions_per_chunk": hyqe_questions_per_chunk,
                "hyqe_temperature": hyqe_temperature,
                "hyqe_question_style": hyqe_question_style,
                "hyqe_quality": hyqe_quality,
                "hyqe_include_chunk_summary": hyqe_include_chunk_summary,
                "show_hyqe_details": show_hyqe_details,
                "clear_chat": clear_chat,
                "reset_db": reset_db
            }
    
    def render_chat_history(self, chat_history: List[tuple]):
        """Render the chat history.
        
        Args:
            chat_history: List of (user_msg, bot_msg, contexts, ner_analysis, expanded_queries, hyde_documents) tuples
        """
        for i, history_item in enumerate(chat_history):
            # Handle different format versions
            if len(history_item) == 3:
                user_msg, bot_msg, contexts = history_item
                ner_analysis = None
                expanded_queries = []
                hyde_documents = []
            elif len(history_item) == 4:
                user_msg, bot_msg, contexts, ner_analysis = history_item
                expanded_queries = []
                hyde_documents = []
            elif len(history_item) == 5:
                user_msg, bot_msg, contexts, ner_analysis, expanded_queries = history_item
                hyde_documents = []
            else:
                user_msg, bot_msg, contexts, ner_analysis, expanded_queries, hyde_documents = history_item
            
            # User message
            with st.chat_message("user"):
                st.write(user_msg)
            
            # Assistant message
            with st.chat_message("assistant"):
                st.write(bot_msg)
                
                # Show HyDE documents if available
                if hyde_documents:
                    with st.expander(f"📝 HyDE Documents ({len(hyde_documents)} generated)"):
                        st.markdown("**Hypothetical documents generated:**")
                        for j, hyde_doc in enumerate(hyde_documents, 1):
                            st.markdown(f"**Document {j}:**")
                            st.markdown(hyde_doc)
                            if j < len(hyde_documents):
                                st.markdown("---")
                
                # Show expanded queries if available
                if expanded_queries:
                    with st.expander(f"🔍 Query Expansion ({len(expanded_queries)} variations)"):
                        st.markdown("**Alternative queries generated:**")
                        for j, expanded_query in enumerate(expanded_queries, 1):
                            st.markdown(f"{j}. {expanded_query}")
                
                # Show contexts in an expander
                if contexts:
                    with st.expander(f"📚 Retrieved Contexts ({len(contexts)} found)"):
                        for j, ctx in enumerate(contexts, 1):
                            metadata = ctx.get("metadata", {})
                            similarity = ctx.get("similarity", 0)
                            query_source = ctx.get("query_source", "unknown")
                            
                            # Show source of context with appropriate emoji
                            if "original" in query_source:
                                if "question" in query_source:
                                    source_indicator = "❓🎯"
                                    source_text = "Original Query → Question"
                                else:
                                    source_indicator = "🎯"
                                    source_text = "Original Query → Document"
                            elif "hyde" in query_source:
                                if "question" in query_source:
                                    source_indicator = "❓📝"
                                    source_text = "HyDE → Question"
                                else:
                                    source_indicator = "📝"
                                    source_text = "HyDE → Document"
                            elif "expanded" in query_source:
                                if "question" in query_source:
                                    source_indicator = "❓🔍"
                                    source_text = "Expanded Query → Question"
                                else:
                                    source_indicator = "🔍"
                                    source_text = "Expanded Query → Document"
                            else:
                                source_indicator = "❓"
                                source_text = "Unknown Source"
                            
                            st.markdown(f"**Context {j}** {source_indicator} {source_text} (Similarity: {similarity:.3f})")
                            
                            # Show source query information
                            if "expanded" in query_source or "hyde" in query_source:
                                source_query = ctx.get("source_query", "")
                                if source_query:
                                    if "expanded" in query_source:
                                        st.markdown(f"*Found via expanded query: {source_query}*")
                                    else:
                                        st.markdown(f"*Found via HyDE document: {source_query[:100]}...*")
                            
                            # Show question or content based on source
                            if "question" in query_source:
                                # This came from a question embedding
                                source_question = metadata.get("source_question") or metadata.get("question", "N/A")
                                st.markdown(f"**Matched Question:** {source_question}")
                                
                                # Show the original content this question was generated from
                                original_content = metadata.get("answer", metadata.get("chunk_content", "N/A"))
                                if len(original_content) > 200:
                                    st.markdown(f"**Related Content:** {original_content[:200]}...")
                                else:
                                    st.markdown(f"**Related Content:** {original_content}")
                            else:
                                # This came from a document embedding
                                st.markdown(f"**Q:** {metadata.get('question', 'N/A')}")
                                
                                answer = metadata.get('answer', 'N/A')
                                if len(answer) > 200:
                                    st.markdown(f"**A:** {answer[:200]}...")
                                else:
                                    st.markdown(f"**A:** {answer}")
                            
                            if j < len(contexts):
                                st.markdown("---")
                
                # Show NER analysis if available
                if ner_analysis and "query" in ner_analysis:
                    query_entities = ner_analysis["query"]
                    if query_entities:
                        with st.expander(f"🧠 NER Analysis ({len(query_entities)} entities found)"):
                            st.markdown("**Entities in your question:**")
                            for entity in query_entities:
                                st.markdown(f"- **{entity.text}** ({entity.label}): {entity.description}")
    
    def render_chat_input(self, placeholder: str = "Ask a question about Python...") -> Optional[str]:
        """Render the chat input and return the user's message.
        
        Args:
            placeholder: Placeholder text for the input
            
        Returns:
            User's message or None
        """
        return st.chat_input(placeholder)
    
    def render_response_with_contexts(self, response: str, contexts: List[Dict[str, Any]]):
        """Render a response with its contexts.
        
        Args:
            response: Generated response
            contexts: Retrieved contexts
        """
        # Display the response
        st.write(response)
        
        # Show contexts
        if contexts:
            with st.expander(f"📚 Retrieved Contexts ({len(contexts)} found)"):
                for i, ctx in enumerate(contexts, 1):
                    metadata = ctx.get("metadata", {})
                    similarity = ctx.get("similarity", 0)
                    
                    # Context header
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Context {i}**")
                    with col2:
                        st.markdown(f"*Similarity: {similarity:.3f}*")
                    
                    # Question
                    st.markdown(f"**Q:** {metadata.get('question', 'N/A')}")
                    
                    # Answer (show more text, no nested interactions)
                    answer = metadata.get('answer', 'N/A')
                    if len(answer) > 500:
                        st.markdown(f"**A:** {answer[:500]}... *(truncated)*")
                    else:
                        st.markdown(f"**A:** {answer}")
                    
                    # Keywords if available
                    keywords = metadata.get('keywords', '')
                    if keywords:
                        st.markdown(f"**Keywords:** {keywords}")
                    
                    if i < len(contexts):
                        st.markdown("---")
    
    def render_response_with_contexts_and_ner(self, response: str, contexts: List[Dict[str, Any]], 
                                             ner_analysis: Dict[str, Any] = None, ner_enabled: bool = False):
        """Render a response with its contexts and NER analysis.
        
        Args:
            response: Generated response
            contexts: Retrieved contexts
            ner_analysis: NER analysis results
            ner_enabled: Whether NER was enabled for this query
        """
        # Display the response
        st.write(response)
        
        # Show contexts
        if contexts:
            with st.expander(f"📚 Retrieved Contexts ({len(contexts)} found)"):
                for i, ctx in enumerate(contexts, 1):
                    metadata = ctx.get("metadata", {})
                    similarity = ctx.get("similarity", 0)
                    
                    # Context header
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Context {i}**")
                    with col2:
                        st.markdown(f"*Similarity: {similarity:.3f}*")
                    
                    # Question
                    st.markdown(f"**Q:** {metadata.get('question', 'N/A')}")
                    
                    # Answer (show more text, no nested interactions)
                    answer = metadata.get('answer', 'N/A')
                    if len(answer) > 500:
                        st.markdown(f"**A:** {answer[:500]}... *(truncated)*")
                    else:
                        st.markdown(f"**A:** {answer}")
                    
                    # Keywords if available
                    keywords = metadata.get('keywords', '')
                    if keywords:
                        st.markdown(f"**Keywords:** {keywords}")
                    
                    # Show NER entities for this context if available
                    if ner_analysis and "contexts" in ner_analysis:
                        try:
                            context_entities = ner_analysis["contexts"][i-1].get("entities", [])
                            if context_entities:
                                entities_text = ", ".join([f"**{e.text}** ({e.label})" for e in context_entities[:5]])
                                st.markdown(f"**Key Entities:** {entities_text}")
                        except (IndexError, KeyError, AttributeError):
                            pass
                    
                    if i < len(contexts):
                        st.markdown("---")
        
        # Show NER analysis if available and enabled
        if ner_enabled and ner_analysis:
            query_entities = ner_analysis.get("query", [])
            if query_entities:
                with st.expander(f"🧠 NER Analysis ({len(query_entities)} entities found in query)"):
                    st.markdown("**Entities identified in your question:**")
                    
                    # Group entities by type
                    entities_by_type = {}
                    for entity in query_entities:
                        entity_type = entity.label
                        if entity_type not in entities_by_type:
                            entities_by_type[entity_type] = []
                        entities_by_type[entity_type].append(entity)
                    
                    # Display entities grouped by type
                    for entity_type, entities in entities_by_type.items():
                        st.markdown(f"**{entity_type}:**")
                        for entity in entities:
                            st.markdown(f"- **{entity.text}**: {entity.description}")
                        st.markdown("")
                    
                    # Show entity summary
                    if len(entities_by_type) > 1:
                        st.markdown("**Summary:**")
                        total_entities = len(query_entities)
                        entity_types = list(entities_by_type.keys())
                        st.markdown(f"Found {total_entities} entities across {len(entity_types)} types: {', '.join(entity_types)}")
    
    def render_response_with_contexts_ner_and_expansion(self, response: str, contexts: List[Dict[str, Any]], 
                                                      ner_analysis: Dict[str, Any] = None, ner_enabled: bool = False,
                                                      expanded_queries: List[str] = None, expansion_enabled: bool = False,
                                                      hyde_documents: List[str] = None, hyde_enabled: bool = False,
                                                      hyqe_enabled: bool = False,
                                                      show_ner_details: bool = False, show_expansion_details: bool = False,
                                                      show_hyde_details: bool = False, show_hyqe_details: bool = False):
        """Render a response with its contexts, NER analysis, query expansion, and HyDE details.
        
        Args:
            response: Generated response
            contexts: Retrieved contexts
            ner_analysis: NER analysis results
            ner_enabled: Whether NER was enabled for this query
            expanded_queries: List of expanded queries
            expansion_enabled: Whether query expansion was enabled
            hyde_documents: List of generated hypothetical documents
            hyde_enabled: Whether HyDE was enabled
            hyqe_enabled: Whether HyQE was enabled
            show_ner_details: Whether to show detailed NER analysis
            show_expansion_details: Whether to show query expansion details
            show_hyde_details: Whether to show HyDE details
            show_hyqe_details: Whether to show HyQE details
        """
        # Display the response
        st.write(response)
        
        # Show HyDE details if enabled and available
        if show_hyde_details and hyde_documents:
            with st.expander(f"📝 HyDE Documents ({len(hyde_documents)} generated)"):
                st.markdown("**Hypothetical documents generated to enhance retrieval:**")
                for i, hyde_doc in enumerate(hyde_documents, 1):
                    st.markdown(f"**Document {i}:**")
                    st.markdown(hyde_doc)
                    if i < len(hyde_documents):
                        st.markdown("---")
        
        # Show HyQE details if enabled and available
        if show_hyqe_details and hyqe_enabled and contexts:
            # Count question-based contexts
            question_contexts = [ctx for ctx in contexts if ctx.get("source") == "question_embedding"]
            if question_contexts:
                with st.expander(f"❓ HyQE Questions ({len(question_contexts)} question matches)"):
                    st.markdown("**Retrieved contexts from hypothetical questions:**")
                    for i, ctx in enumerate(question_contexts, 1):
                        metadata = ctx.get("metadata", {})
                        similarity = ctx.get("similarity", 0)
                        source_question = metadata.get("source_question", metadata.get("question", "N/A"))
                        chunk_content = metadata.get("chunk_content", metadata.get("answer", "N/A"))
                        
                        st.markdown(f"**Question {i}:** {source_question}")
                        st.markdown(f"*Similarity: {similarity:.3f}*")
                        if chunk_content != "N/A" and len(chunk_content) > 200:
                            st.markdown(f"*Source chunk: {chunk_content[:200]}...*")
                        else:
                            st.markdown(f"*Source chunk: {chunk_content}*")
                        
                        if i < len(question_contexts):
                            st.markdown("---")
        
        # Show query expansion details if enabled and available
        if show_expansion_details and expanded_queries:
            with st.expander(f"🔍 Query Expansion ({len(expanded_queries)} variations generated)"):
                st.markdown("**Alternative queries used for enhanced search:**")
                for i, expanded_query in enumerate(expanded_queries, 1):
                    st.markdown(f"{i}. {expanded_query}")
        
        # Show contexts
        if contexts:
            with st.expander(f"📚 Retrieved Contexts ({len(contexts)} found)"):
                for i, ctx in enumerate(contexts, 1):
                    metadata = ctx.get("metadata", {})
                    similarity = ctx.get("similarity", 0)
                    query_source = ctx.get("query_source", "unknown")
                    
                    # Context header with source indicator
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        # Enhanced source indicators
                        if ctx.get("source") == "question_embedding":
                            source_indicator = "❓ Question Match"
                        elif query_source == "original":
                            source_indicator = "🎯 Original"
                        elif "hyde" in query_source:
                            source_indicator = "📝 HyDE"
                        else:
                            source_indicator = "🔍 Expanded"
                        st.markdown(f"**Context {i}** ({source_indicator})")
                    with col2:
                        st.markdown(f"*Similarity: {similarity:.3f}*")
                    
                    # Show source query if from expansion
                    if query_source == "expanded" and show_expansion_details:
                        source_query = ctx.get("source_query", "")
                        if source_query:
                            st.markdown(f"*Found via: {source_query}*")
                    
                    # Display based on source type
                    if ctx.get("source") == "question_embedding":
                        # Question-based result
                        source_question = metadata.get("source_question", "N/A")
                        chunk_content = metadata.get("chunk_content", "N/A")
                        
                        st.markdown(f"**Generated Question:** {source_question}")
                        st.markdown(f"**Source Content:** {chunk_content}")
                    else:
                        # Document-based result
                        st.markdown(f"**Q:** {metadata.get('question', 'N/A')}")
                        
                        # Answer (show more text, no nested interactions)
                        answer = metadata.get('answer', 'N/A')
                        if len(answer) > 500:
                            st.markdown(f"**A:** {answer[:500]}... *(truncated)*")
                        else:
                            st.markdown(f"**A:** {answer}")
                        
                        # Keywords if available
                        keywords = metadata.get('keywords', '')
                        if keywords:
                            st.markdown(f"**Keywords:** {keywords}")
                    
                    # Show NER entities for this context if available and requested
                    if show_ner_details and ner_analysis and "contexts" in ner_analysis:
                        try:
                            context_entities = ner_analysis["contexts"][i-1].get("entities", [])
                            if context_entities:
                                entities_text = ", ".join([f"**{e.text}** ({e.label})" for e in context_entities[:5]])
                                st.markdown(f"**Key Entities:** {entities_text}")
                        except (IndexError, KeyError, AttributeError):
                            pass
                    
                    if i < len(contexts):
                        st.markdown("---")
        
        # Show detailed NER analysis if enabled
        if show_ner_details and ner_analysis and "query" in ner_analysis:
            query_entities = ner_analysis["query"]
            if query_entities:
                with st.expander(f"🧠 NER Analysis ({len(query_entities)} entities found)"):
                    st.markdown("**Entities in your question:**")
                    
                    # Group entities by type
                    entities_by_type = {}
                    for entity in query_entities:
                        entity_type = entity.label
                        if entity_type not in entities_by_type:
                            entities_by_type[entity_type] = []
                        entities_by_type[entity_type].append(entity)
                    
                    # Display grouped entities
                    for entity_type, entities in entities_by_type.items():
                        st.markdown(f"**{entity_type}:**")
                        for entity in entities:
                            st.markdown(f"- **{entity.text}**: {entity.description}")
                        st.markdown("")
                    
                    # Show entity summary
                    if len(entities_by_type) > 1:
                        st.markdown("**Summary:**")
                        total_entities = len(query_entities)
                        entity_types = list(entities_by_type.keys())
                        st.markdown(f"Found {total_entities} entities across {len(entity_types)} types: {', '.join(entity_types)}")
    
    def show_initialization_progress(self, step: str, progress: float = None):
        """Show initialization progress.
        
        Args:
            step: Current step description
            progress: Progress value (0-1) or None for spinner
        """
        if progress is not None:
            st.progress(progress)
        else:
            with st.spinner(step):
                pass
    
    def show_error(self, error_message: str):
        """Show an error message.
        
        Args:
            error_message: Error message to display
        """
        st.error(error_message)
    
    def show_success(self, message: str):
        """Show a success message.
        
        Args:
            message: Success message to display
        """
        st.success(message)
    
    def show_info(self, message: str):
        """Show an info message.
        
        Args:
            message: Info message to display
        """
        st.info(message)
    
    def show_warning(self, message: str):
        """Show a warning message.
        
        Args:
            message: Warning message to display
        """
        st.warning(message)
    
    def render_debug_info(self, debug_data: Dict[str, Any]):
        """Render debug information.
        
        Args:
            debug_data: Debug data to display
        """
        with st.expander("🐛 Debug Information"):
            st.json(debug_data)
    
    def render_metrics(self, metrics: Dict[str, Any]):
        """Render metrics in columns.
        
        Args:
            metrics: Dictionary of metrics to display
        """
        cols = st.columns(len(metrics))
        
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(key, value)
