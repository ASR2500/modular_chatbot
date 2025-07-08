"""Main application module for the Python FAQ RAG chatbot."""

import streamlit as st
import logging
from typing import Dict, Any

from src.config import Config
from src.data.processor import DataProcessor
from src.rag.database import DatabaseManager
from src.rag.engine import RAGEngine
from src.ui.components import ChatbotUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PythonFAQChatbot:
    """Main chatbot application."""
    
    def __init__(self):
        """Initialize the chatbot application."""
        self.ui = ChatbotUI()
    
    @property
    def data_processor(self):
        """Get data processor from session state."""
        return st.session_state.get('data_processor', None)
    
    @property
    def db_manager(self):
        """Get database manager from session state."""
        return st.session_state.get('db_manager', None)
    
    @property
    def rag_engine(self):
        """Get RAG engine from session state."""
        return st.session_state.get('rag_engine', None)
    
    @property
    def initialized(self):
        """Check if components are initialized."""
        return st.session_state.get('initialization_complete', False)
    
    def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return
        
        try:
            # Validate configuration
            Config.validate()
            
            # Initialize components and store in session state
            st.session_state.data_processor = DataProcessor()
            st.session_state.db_manager = DatabaseManager()
            st.session_state.rag_engine = RAGEngine(st.session_state.db_manager)
            
            # Load and process data
            self.setup_data()
            
            # Mark as initialized in session state
            st.session_state.initialization_complete = True
            logger.info("Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {str(e)}")
            raise
    
    def setup_data(self):
        """Set up the data and vector database."""
        # Load data
        faq_data = self.data_processor.load_and_process_data()
        
        # Get or create collection
        collection = self.db_manager.get_or_create_collection()
        
        # Populate collection if empty
        if collection.count() == 0:
            documents = self.data_processor.prepare_documents_for_embedding(faq_data)
            
            # Create progress callback
            def progress_callback(progress):
                if hasattr(st, 'session_state') and 'progress_bar' in st.session_state:
                    st.session_state.progress_bar.progress(progress)
            
            self.db_manager.populate_collection(
                documents=documents,
                progress_callback=progress_callback
            )
        
        # Set up question embeddings if HyQE is enabled
        if self.rag_engine.enable_hyqe:
            questions_collection = self.db_manager.get_or_create_questions_collection()
            
            # Generate questions if collection is empty
            if questions_collection.count() == 0:
                st.info("🔍 Generating hypothetical questions for enhanced retrieval...")
                
                # Get documents for question generation
                documents = self.data_processor.prepare_documents_for_embedding(faq_data)
                
                # Create progress bar for question generation
                question_progress_bar = st.progress(0)
                
                def question_progress_callback(progress):
                    question_progress_bar.progress(progress)
                
                # Generate and populate question embeddings
                success = self.rag_engine.generate_and_populate_question_embeddings(
                    documents=documents,
                    progress_callback=question_progress_callback
                )
                
                if success:
                    st.success(f"✅ Generated hypothetical questions for enhanced retrieval!")
                else:
                    st.warning("⚠️ Failed to generate question embeddings, proceeding with document embeddings only")
            else:
                logger.info(f"Using existing questions collection with {questions_collection.count()} questions")
    
    def run(self):
        """Run the Streamlit application."""
        # Initialize session state
        self.init_session_state()
        
        # Render header
        self.ui.render_header()
        
        # Initialize components if not already done
        if not self.initialized:
            self.initialize_with_ui()
        
        # Get data and engine stats (with safety checks)
        data_stats = {}
        engine_stats = {}
        
        if self.data_processor is not None:
            try:
                data_stats = self.data_processor.get_data_stats()
            except Exception as e:
                logger.error(f"Error getting data stats: {e}")
                data_stats = {"error": str(e)}
        
        if self.rag_engine is not None:
            try:
                engine_stats = self.rag_engine.get_engine_stats()
            except Exception as e:
                logger.error(f"Error getting engine stats: {e}")
                engine_stats = {"error": str(e)}
        
        # Render sidebar and get user settings
        user_settings = self.ui.render_sidebar(data_stats, engine_stats)
        
        # Handle user actions
        self.handle_user_actions(user_settings)
        
        # Render chat interface
        self.render_chat_interface(user_settings)
    
    def init_session_state(self):
        """Initialize Streamlit session state."""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "initialization_complete" not in st.session_state:
            st.session_state.initialization_complete = False
    
    def initialize_with_ui(self):
        """Initialize components with UI feedback."""
        if st.session_state.get('initialization_complete', False):
            return
        
        try:
            with st.spinner("🔧 Initializing chatbot components..."):
                # Validate configuration
                Config.validate()
                
                # Initialize components and store in session state
                st.session_state.data_processor = DataProcessor()
                st.session_state.db_manager = DatabaseManager()
                st.session_state.rag_engine = RAGEngine(st.session_state.db_manager)
            
            with st.spinner("📊 Loading and processing FAQ data..."):
                faq_data = st.session_state.data_processor.load_and_process_data()
                self.ui.show_success(f"Loaded {len(faq_data)} FAQ entries")
            
            with st.spinner("🗄️ Setting up vector database..."):
                collection = st.session_state.db_manager.get_or_create_collection()
                
                if collection.count() == 0:
                    self.ui.show_info("Populating vector database with FAQ data...")
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    st.session_state.progress_bar = progress_bar
                    
                    # Prepare and populate
                    documents = st.session_state.data_processor.prepare_documents_for_embedding(faq_data)
                    
                    def progress_callback(progress):
                        progress_bar.progress(progress)
                    
                    st.session_state.db_manager.populate_collection(
                        documents=documents,
                        progress_callback=progress_callback
                    )
                    
                    self.ui.show_success("Vector database populated successfully!")
                else:
                    self.ui.show_info(f"Using existing vector database with {collection.count()} documents")
            
            # Set up question embeddings if HyQE is enabled
            if st.session_state.rag_engine.enable_hyqe:
                with st.spinner("❓ Setting up hypothetical questions (HyQE)..."):
                    questions_collection = st.session_state.db_manager.get_or_create_questions_collection()
                    
                    # Generate questions if collection is empty
                    if questions_collection.count() == 0:
                        self.ui.show_info("🔍 Generating hypothetical questions for enhanced retrieval...")
                        
                        # Get documents for question generation
                        documents = st.session_state.data_processor.prepare_documents_for_embedding(faq_data)
                        
                        # Create progress bar for question generation
                        question_progress_bar = st.progress(0)
                        
                        def question_progress_callback(progress):
                            question_progress_bar.progress(progress)
                        
                        # Generate and populate question embeddings
                        success = st.session_state.rag_engine.generate_and_populate_question_embeddings(
                            documents=documents,
                            progress_callback=question_progress_callback
                        )
                        
                        if success:
                            self.ui.show_success(f"✅ Generated hypothetical questions for enhanced retrieval!")
                            questions_count = questions_collection.count()
                            self.ui.show_info(f"📊 HyQE collection now has {questions_count} question embeddings")
                        else:
                            self.ui.show_error("⚠️ Failed to generate question embeddings, proceeding with document embeddings only")
                    else:
                        questions_count = questions_collection.count()
                        self.ui.show_info(f"Using existing HyQE collection with {questions_count} question embeddings")
            else:
                self.ui.show_info("HyQE is disabled - using document embeddings only")
            
            # Mark initialization as complete
            st.session_state.initialization_complete = True
            
        except Exception as e:
            self.ui.show_error(f"Failed to initialize: {str(e)}")
            st.stop()
    
    def handle_user_actions(self, user_settings: Dict[str, Any]):
        """Handle user actions from the sidebar.
        
        Args:
            user_settings: User settings from sidebar
        """
        if user_settings.get("clear_chat"):
            st.session_state.chat_history = []
            self.ui.show_success("Chat history cleared!")
            st.rerun()
        
        if user_settings.get("reset_db"):
            try:
                with st.spinner("Resetting vector database..."):
                    if self.db_manager:
                        self.db_manager.reset_collection()
                    st.session_state.initialization_complete = False
                    # Clear all session state objects
                    for key in ['data_processor', 'db_manager', 'rag_engine']:
                        if key in st.session_state:
                            del st.session_state[key]
                
                self.ui.show_success("Vector database reset! Please refresh the page.")
                st.stop()
                
            except Exception as e:
                self.ui.show_error(f"Failed to reset database: {str(e)}")
    
    def render_chat_interface(self, user_settings: Dict[str, Any]):
        """Render the chat interface.
        
        Args:
            user_settings: User settings from sidebar
        """
        # Chat header
        st.header("💬 Chat")
        
        # Check if RAG engine is available
        if not self.initialized or self.rag_engine is None:
            st.warning("🔄 Chatbot is still initializing. Please wait...")
            return
        
        # Display chat history
        self.ui.render_chat_history(st.session_state.chat_history)
        
        # Chat input
        if prompt := self.ui.render_chat_input():
            # Add user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching FAQ database and generating response..."):
                    try:
                        # Process the query with NER, Query Expansion, HyDE, and HyQE
                        use_ner = user_settings.get("enable_ner", True)
                        use_query_expansion = user_settings.get("enable_query_expansion", True)
                        num_expanded_queries = user_settings.get("num_expanded_queries", 3)
                        show_expansion_details = user_settings.get("show_expansion_details", False)
                        
                        # HyDE settings
                        use_hyde = user_settings.get("enable_hyde", True)
                        show_hyde_details = user_settings.get("show_hyde_details", False)
                        
                        # HyQE settings
                        use_hyqe = user_settings.get("enable_hyqe", True)
                        show_hyqe_details = user_settings.get("show_hyqe_details", False)
                        
                        # Configure HyDE processor if enabled
                        if use_hyde and self.rag_engine.hyde_processor:
                            # Update HyDE config dynamically
                            hyde_config = self.rag_engine.hyde_processor.config
                            hyde_config.TEMPERATURE = user_settings.get("hyde_temperature", 0.2)
                            hyde_config.MAX_TOKENS = user_settings.get("hyde_max_tokens", 150)
                            hyde_config.ANSWER_STYLE = user_settings.get("hyde_answer_style", "concise")
                            hyde_config.INCLUDE_EXAMPLES = user_settings.get("hyde_include_examples", True)
                            hyde_config.BATCH_MODE = user_settings.get("hyde_batch_mode", False)
                            hyde_config.BATCH_SIZE = user_settings.get("hyde_batch_size", 3)
                        
                        # Configure HyQE processor if enabled
                        if use_hyqe and self.rag_engine.hyqe_processor:
                            # Update HyQE config dynamically
                            hyqe_config = self.rag_engine.hyqe_processor.config
                            hyqe_config.QUESTIONS_PER_CHUNK = user_settings.get("hyqe_questions_per_chunk", 3)
                            hyqe_config.TEMPERATURE = user_settings.get("hyqe_temperature", 0.4)
                            hyqe_config.QUESTION_STYLE = user_settings.get("hyqe_question_style", "natural")
                            hyqe_config.QUALITY = user_settings.get("hyqe_quality", "balanced")
                            hyqe_config.INCLUDE_CHUNK_SUMMARY = user_settings.get("hyqe_include_chunk_summary", True)
                        
                        # Process query with all enhancements
                        contexts = self.rag_engine.retrieve_contexts(
                            query=prompt,
                            n_results=user_settings.get("n_results", Config.DEFAULT_N_RESULTS),
                            use_query_expansion=use_query_expansion,
                            num_expanded_queries=num_expanded_queries,
                            use_hyde=use_hyde,
                            use_hyqe=use_hyqe
                        )
                        
                        # Generate response
                        response = self.rag_engine.generate_response(prompt, contexts)
                        
                        # Get additional details for display
                        expanded_queries = []
                        hyde_documents = []
                        ner_analysis = None
                        
                        if show_expansion_details and use_query_expansion and self.rag_engine.query_expansion_processor:
                            try:
                                expanded_queries = self.rag_engine.query_expansion_processor.expand_query(
                                    prompt, num_expansions=num_expanded_queries
                                )
                            except Exception as e:
                                logger.warning(f"Failed to get expansion details: {e}")
                        
                        if show_hyde_details and use_hyde and self.rag_engine.hyde_processor:
                            try:
                                hyde_documents = self.rag_engine.hyde_processor.generate_hypothetical_documents(
                                    prompt, domain="python"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to get HyDE details: {e}")
                        
                        if use_ner and self.rag_engine.ner_processor:
                            try:
                                ner_analysis = self.rag_engine.ner_processor.analyze_entities(prompt)
                            except Exception as e:
                                logger.warning(f"Failed to get NER analysis: {e}")
                        
                        # Render response with all enhancements
                        self.ui.render_response_with_contexts_ner_and_expansion(
                            response=response,
                            contexts=contexts,
                            ner_analysis=ner_analysis,
                            ner_enabled=use_ner,
                            expanded_queries=expanded_queries,
                            expansion_enabled=use_query_expansion,
                            hyde_documents=hyde_documents,
                            hyde_enabled=use_hyde,
                            hyqe_enabled=use_hyqe,
                            show_ner_details=user_settings.get("show_ner_details", False),
                            show_expansion_details=show_expansion_details,
                            show_hyde_details=show_hyde_details,
                            show_hyqe_details=show_hyqe_details
                        )
                        
                        # Add to chat history (including all enhancement data)
                        st.session_state.chat_history.append((
                            prompt,
                            response,
                            contexts,
                            ner_analysis,
                            expanded_queries,
                            hyde_documents
                        ))
                        
                    except Exception as e:
                        error_msg = f"Error processing query: {str(e)}"
                        logger.error(error_msg)
                        self.ui.show_error(error_msg)

def main():
    """Main entry point for the application."""
    try:
        # Create and run the chatbot
        chatbot = PythonFAQChatbot()
        chatbot.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
