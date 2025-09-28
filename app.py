"""
Main Streamlit application for the arXiv Paper Explainer and Q&A System.
FINAL VERSION - All file handling issues resolved.
"""

import streamlit as st
import time
import os
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import our custom modules
from config import Config
from paper_processor import PaperProcessor
from rag_system import RAGSystem
from utils import (
    setup_page_config, 
    add_custom_css, 
    display_error, 
    display_success,
    format_file_size,
    truncate_text
)

class PaperExplainerApp:
    """Main application class for the Paper Explainer."""
    
    def __init__(self):
        """Initialize the application."""
        self.config = Config()
        
        # Initialize session state first
        self._initialize_session_state()
        
        # Initialize paper processor
        if 'paper_processor' not in st.session_state:
            st.session_state.paper_processor = PaperProcessor()
        
        self.paper_processor = st.session_state.paper_processor
        
        # Initialize RAG system only once and store in session state
        if 'rag_system' not in st.session_state:
            with st.spinner("Initializing RAG system..."):
                try:
                    st.session_state.rag_system = RAGSystem()
                    if st.session_state.rag_system._initialization_complete:
                        st.success("RAG system initialized successfully!")
                    else:
                        st.error("RAG system initialization failed")
                        st.stop()
                except Exception as e:
                    st.error(f"Failed to initialize RAG system: {str(e)}")
                    st.stop()
        
        self.rag_system = st.session_state.rag_system
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        default_values = {
            'paper_processed': False,
            'paper_metadata': None,
            'paper_text': None,
            'chat_history': [],
            'processing_start_time': None,
            'rag_system_ready': False,
            'current_paper_data': None,
            'last_arxiv_url': None,
            'last_uploaded_file': None,
            'processing_in_progress': False
        }
        
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Run the main application."""
        # Set up page configuration and styling
        setup_page_config()
        add_custom_css()
        
        # Validate configuration
        try:
            self.config.validate_config()
        except ValueError as e:
            st.error(f"Configuration Error: {str(e)}")
            st.info("Please check your .env file and ensure GROQ_API_KEY is set.")
            st.stop()
        
        # Display header
        self._display_header()
        
        # Main application layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._display_sidebar()
        
        with col2:
            self._display_main_content()
    
    def _display_header(self):
        """Display the application header."""
        st.markdown('<div class="main-header">arXiv Paper Explainer & Q&A System</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-powered research paper analysis using Retrieval-Augmented Generation</div>', unsafe_allow_html=True)
        
        # Display system status
        if st.session_state.paper_processed and st.session_state.rag_system_ready:
            st.success("Paper loaded and ready for questions!")
        elif st.session_state.processing_in_progress:
            st.warning("Processing paper... Please wait.")
        elif st.session_state.paper_processed:
            st.warning("Paper loaded but setting up RAG system...")
        else:
            st.info("Upload a paper or provide an arXiv URL to get started")
    
    def _display_sidebar(self):
        """Display the sidebar with paper upload/processing options."""
        st.header("Paper Input")
        
        # Prevent input while processing
        input_disabled = st.session_state.processing_in_progress
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["arXiv URL", "Upload PDF"],
            help="Select how you want to provide the research paper",
            disabled=input_disabled
        )
        
        paper_data = None
        
        if not input_disabled:
            if input_method == "arXiv URL":
                paper_data = self._handle_arxiv_input()
            else:
                paper_data = self._handle_pdf_upload()
        
        # Store paper data in session state
        if paper_data:
            st.session_state.current_paper_data = paper_data
        
        # Process paper button - only show if we have data and it's not already processed
        if st.session_state.current_paper_data and not st.session_state.paper_processed and not input_disabled:
            if st.button("Process Paper", type="primary", key="process_button"):
                self._process_paper(st.session_state.current_paper_data)
        
        # Reset button if paper is already processed
        if st.session_state.paper_processed and not input_disabled:
            if st.button("Process New Paper", type="secondary", key="reset_button"):
                self._reset_application()
        
        # Display paper info if processed
        if st.session_state.paper_processed and st.session_state.paper_metadata:
            self._display_paper_info()
        
        # Display processing statistics
        if st.session_state.paper_processed:
            self._display_processing_stats()
        
        # Debug information
        if st.session_state.current_paper_data:
            with st.expander("Debug Info", expanded=False):
                st.json({
                    'file_exists': os.path.exists(st.session_state.current_paper_data.get('local_path', '')),
                    'file_path': st.session_state.current_paper_data.get('local_path', ''),
                    'file_size': st.session_state.current_paper_data.get('file_size', 0)
                })
    
    def _reset_application(self):
        """Reset the application to process a new paper."""
        # Cleanup old files
        if hasattr(self.paper_processor, 'cleanup_temp_files'):
            self.paper_processor.cleanup_temp_files()
        
        # Reset session state
        st.session_state.paper_processed = False
        st.session_state.paper_metadata = None
        st.session_state.paper_text = None
        st.session_state.chat_history = []
        st.session_state.rag_system_ready = False
        st.session_state.current_paper_data = None
        st.session_state.last_arxiv_url = None
        st.session_state.last_uploaded_file = None
        st.session_state.processing_in_progress = False
        
        # Reset RAG system
        if hasattr(self.rag_system, 'reset_system'):
            self.rag_system.reset_system()
        
        # Create new paper processor
        st.session_state.paper_processor = PaperProcessor()
        self.paper_processor = st.session_state.paper_processor
        
        st.success("Application reset! You can now process a new paper.")
        st.rerun()
    
    def _handle_arxiv_input(self) -> Optional[dict]:
        """Handle arXiv URL input."""
        st.subheader("arXiv Paper URL")
        
        # Sample URLs for quick testing
        sample_urls = {
            "Attention Is All You Need": "https://arxiv.org/abs/1706.03762",
            "ResNet": "https://arxiv.org/abs/1512.03385",
            "BERT": "https://arxiv.org/abs/1810.04805"
        }
        
        # Quick select dropdown
        selected_sample = st.selectbox(
            "Or choose a sample paper:",
            ["Custom URL"] + list(sample_urls.keys()),
            help="Select a famous paper for quick testing"
        )
        
        if selected_sample != "Custom URL":
            arxiv_url = sample_urls[selected_sample]
            st.info(f"Selected: {selected_sample}")
        else:
            arxiv_url = st.text_input(
                "Enter arXiv URL:",
                placeholder="https://arxiv.org/abs/1706.03762",
                help="Paste the arXiv URL of the paper you want to analyze"
            )
        
        if arxiv_url and arxiv_url.strip():
            # Only download if we don't have this paper already or if it failed before
            if (st.session_state.last_arxiv_url != arxiv_url or 
                not st.session_state.current_paper_data or
                not st.session_state.current_paper_data.get('file_exists', False)):
                
                try:
                    with st.spinner("Downloading paper..."):
                        paper_data = self.paper_processor.download_arxiv_paper(arxiv_url)
                        
                    if paper_data and paper_data.get('file_exists', False):
                        st.session_state.last_arxiv_url = arxiv_url
                        return paper_data
                    else:
                        st.error("Failed to download paper. Please check the URL and try again.")
                        return None
                        
                except Exception as e:
                    st.error(f"Error downloading paper: {str(e)}")
                    return None
            else:
                # Return cached data if available and valid
                if (st.session_state.current_paper_data and 
                    st.session_state.current_paper_data.get('file_exists', False)):
                    return st.session_state.current_paper_data
        
        return None
    
    def _handle_pdf_upload(self) -> Optional[dict]:
        """Handle PDF file upload."""
        st.subheader("Upload PDF File")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help=f"Upload a research paper PDF (max {self.config.MAX_FILE_SIZE_MB}MB)"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size = len(uploaded_file.getvalue())
            st.info(f"File: {uploaded_file.name} ({format_file_size(file_size)})")
            
            # Only process if it's a new file or previous processing failed
            if (st.session_state.last_uploaded_file != uploaded_file.name or 
                not st.session_state.current_paper_data or
                not st.session_state.current_paper_data.get('file_exists', False)):
                
                try:
                    paper_data = self.paper_processor.process_uploaded_pdf(uploaded_file)
                    
                    if paper_data and paper_data.get('file_exists', False):
                        st.session_state.last_uploaded_file = uploaded_file.name
                        return paper_data
                    else:
                        st.error("Failed to process uploaded file.")
                        return None
                        
                except Exception as e:
                    st.error(f"Error processing uploaded file: {str(e)}")
                    return None
            else:
                # Return cached data if available and valid
                if (st.session_state.current_paper_data and 
                    st.session_state.current_paper_data.get('file_exists', False)):
                    return st.session_state.current_paper_data
        
        return None
    
    def _process_paper(self, paper_data: dict):
        """Process the paper and set up RAG system with comprehensive error handling."""
        if not paper_data:
            st.error("No paper data available for processing")
            return
        
        # Verify file still exists
        pdf_path = paper_data.get('local_path')
        if not pdf_path or not os.path.exists(pdf_path):
            st.error("PDF file not found. Please re-upload or re-download the paper.")
            return
        
        # Set processing flag
        st.session_state.processing_in_progress = True
        
        # Reset processing state
        st.session_state.paper_processed = False
        st.session_state.rag_system_ready = False
        st.session_state.chat_history = []
        st.session_state.processing_start_time = time.time()
        
        # Create a container for the processing status
        processing_container = st.container()
        
        with processing_container:
            try:
                # Step 1: Extract text from PDF
                st.info("Step 1: Extracting text from PDF...")
                
                text = self.paper_processor.extract_text_from_pdf(pdf_path)
                if not text or len(text.strip()) < 100:
                    st.error("Failed to extract sufficient text from PDF")
                    return
                
                st.success(f"Extracted {len(text)} characters from PDF")
                
                # Step 2: Process with RAG system
                st.info("Step 2: Setting up RAG system - this may take a few minutes...")
                
                # Process the paper with the RAG system
                success = self.rag_system.process_paper(text, paper_data)
                
                if not success:
                    st.error("Failed to process paper with RAG system")
                    return
                
                # Step 3: Verify system is ready
                if not self.rag_system.is_ready():
                    st.error("RAG system setup failed - system not ready")
                    return
                
                # Step 4: Update session state
                st.session_state.paper_processed = True
                st.session_state.rag_system_ready = True
                st.session_state.paper_metadata = paper_data
                st.session_state.paper_text = text
                
                # Show final success message
                processing_time = time.time() - st.session_state.processing_start_time
                st.success(f"Paper processed successfully in {processing_time:.1f} seconds!")
                
                # Force UI refresh
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing paper: {str(e)}")
                st.error("Please try again or contact support if the issue persists.")
                
            finally:
                # Always clear processing flag
                st.session_state.processing_in_progress = False
    
    def _display_paper_info(self):
        """Display information about the processed paper."""
        st.subheader("Paper Information")
        
        metadata = st.session_state.paper_metadata
        if metadata:
            # Create an attractive info card
            with st.container():
                st.markdown(f"""
                <div class="paper-info">
                    <h4>{metadata['title']}</h4>
                    <p><strong>Authors:</strong> {', '.join(metadata['authors'][:3])}{'...' if len(metadata['authors']) > 3 else ''}</p>
                    <p><strong>Published:</strong> {metadata['published']}</p>
                    <p><strong>Categories:</strong> {', '.join(metadata['categories'][:2])}{'...' if len(metadata['categories']) > 2 else ''}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def _display_processing_stats(self):
        """Display processing and system statistics."""
        st.subheader("Processing Stats")
        
        if st.session_state.paper_text and hasattr(self, 'rag_system'):
            stats = self.paper_processor.get_paper_statistics(st.session_state.paper_text)
            rag_stats = self.rag_system.get_system_stats()
            
            # Create metrics display
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Pages", f"~{stats['word_count'] // 500}")
                st.metric("Words", f"{stats['word_count']:,}")
                st.metric("Characters", f"{stats['character_count']:,}")
            
            with col2:
                st.metric("Chunks", rag_stats.get('total_chunks', 0))
                st.metric("Read Time", f"{stats['estimated_reading_time']:.1f} min")
                st.metric("System Ready", "Yes" if rag_stats.get('system_ready', False) else "No")
    
    def _display_main_content(self):
        """Display the main content area with Q&A interface."""
        if not st.session_state.paper_processed or not st.session_state.rag_system_ready:
            self._display_welcome_screen()
        else:
            self._display_qa_interface()
    
    def _display_welcome_screen(self):
        """Display welcome screen when no paper is loaded."""
        st.header("Welcome to arXiv Paper Explainer!")
        
        st.markdown("""
        ### How it works:
        
        1. **Input**: Provide an arXiv URL or upload a PDF research paper
        2. **Process**: Our AI extracts and indexes the content using advanced RAG technology
        3. **Ask**: Ask any questions about the paper and get intelligent, context-aware answers
        4. **Learn**: Understand complex research papers quickly and efficiently
        
        ### What you can ask:
        - "What is the main contribution of this paper?"
        - "Explain the methodology in simple terms"
        - "What are the key findings and results?"
        - "How does this compare to previous work?"
        - "What are the limitations of this study?"
        
        ### Powered by:
        - **Groq**: Ultra-fast LLM inference with Llama 3.1
        - **FAISS**: Efficient vector storage and retrieval
        - **LangChain**: Advanced RAG pipeline orchestration
        """)
        
        # Display some sample papers
        st.subheader("Try these popular papers:")
        
        sample_papers = [
            ("Attention Is All You Need", "The transformer architecture that revolutionized NLP", "https://arxiv.org/abs/1706.03762"),
            ("Deep Residual Learning", "ResNet - solving the vanishing gradient problem", "https://arxiv.org/abs/1512.03385"),
            ("BERT", "Bidirectional Encoder Representations from Transformers", "https://arxiv.org/abs/1810.04805"),
        ]
        
        for title, desc, url in sample_papers:
            with st.expander(f"{title}"):
                st.write(desc)
                st.code(url)
    
    def _display_qa_interface(self):
        """Display the question-answering interface."""
        st.header("Ask Questions About the Paper")
        
        # Create tabs for different interfaces
        tab1, tab2, tab3 = st.tabs(["Q&A Chat", "Paper Summary", "Search Content"])
        
        with tab1:
            self._display_chat_interface()
        
        with tab2:
            self._display_summary_tab()
        
        with tab3:
            self._display_search_tab()
    
    def _display_chat_interface(self):
        """Display the chat interface for Q&A."""
        # Double-check system is ready
        if not self.rag_system.is_ready():
            st.error("RAG system not ready. Please process a paper first.")
            if st.button("Debug System Status"):
                stats = self.rag_system.get_system_stats()
                st.json(stats)
            return
        
        # Suggested questions
        st.subheader("Suggested Questions")
        suggested_questions = self.rag_system.get_suggested_questions()
        
        # Display suggested questions as buttons
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions[:6]):  # Show first 6
            with cols[i % 2]:
                if st.button(f"{truncate_text(question, 50)}", key=f"suggest_{i}"):
                    self._ask_question(question)
        
        # Chat input
        st.subheader("Ask Your Question")
        
        with st.form("question_form"):
            user_question = st.text_area(
                "What would you like to know about this paper?",
                height=100,
                placeholder="e.g., What is the main contribution of this paper?"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                ask_button = st.form_submit_button("Ask Question", type="primary")
            with col2:
                clear_button = st.form_submit_button("Clear History")
        
        # Handle form submissions
        if ask_button and user_question.strip():
            self._ask_question(user_question)
        
        if clear_button:
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        # Display chat history
        self._display_chat_history()
    
    def _ask_question(self, question: str):
        """Process and answer a user question."""
        if not self.rag_system.is_ready():
            st.error("RAG system not ready to answer questions")
            return
        
        with st.spinner("Processing your question..."):
            try:
                start_time = time.time()
                answer, sources = self.rag_system.ask_question(question)
                response_time = time.time() - start_time
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': answer,
                    'sources': sources,
                    'timestamp': time.time(),
                    'response_time': response_time
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
    
    def _display_chat_history(self):
        """Display the chat history."""
        if not st.session_state.chat_history:
            st.info("No questions asked yet. Try one of the suggested questions above!")
            return
        
        st.subheader("Conversation History")
        
        # Display messages in reverse order (newest first)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                # Question
                st.markdown(f"""
                <div class="chat-message">
                    <strong>Question:</strong> {chat['question']}
                </div>
                """, unsafe_allow_html=True)
                
                # Answer
                st.markdown(f"""
                <div class="chat-message">
                    <strong>Answer:</strong><br>{chat['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Response time and sources
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.caption(f"Response time: {chat['response_time']:.2f}s")
                with col2:
                    st.caption(f"Sources used: {len(chat['sources'])} chunks")
                
                st.divider()
    
    def _display_summary_tab(self):
        """Display paper summary tab."""
        st.subheader("Paper Summary")
        
        if not self.rag_system.is_ready():
            st.error("RAG system not ready. Please process a paper first.")
            return
        
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Generating comprehensive summary..."):
                try:
                    summary = self.rag_system.get_paper_summary()
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
        else:
            st.info("Click the button above to generate an AI-powered summary of the paper.")
    
    def _display_search_tab(self):
        """Display content search tab."""
        st.subheader("Search Paper Content")
        
        if not self.rag_system.is_ready():
            st.error("RAG system not ready. Please process a paper first.")
            return
        
        search_query = st.text_input(
            "Search for specific content:",
            placeholder="e.g., transformer architecture, attention mechanism"
        )
        
        if search_query:
            with st.spinner("Searching..."):
                try:
                    results = self.rag_system.search_paper_content(search_query)
                    
                    if results:
                        st.success(f"Found {len(results)} relevant sections")
                        
                        for i, doc in enumerate(results):
                            with st.expander(f"Result {i+1}"):
                                st.write(doc.page_content)
                                st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    else:
                        st.warning("No relevant content found for your search query.")
                except Exception as e:
                    st.error(f"Error searching content: {str(e)}")

def main():
    """Main function to run the application."""
    try:
        app = PaperExplainerApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page or check your configuration.")
        
        # Show debug info
        if st.button("Show Debug Info"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()