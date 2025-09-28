import os
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import ssl
import urllib3
import tempfile
import time

# Disable SSL warnings and configure SSL context for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from config import Config
from utils import display_error, display_success, format_response_with_sources

class RAGSystem:
    """
    Retrieval-Augmented Generation system for answering questions about research papers.
    Uses FAISS for vector storage with proper LangChain embeddings.
    """
    
    def __init__(self):
        """Initialize the RAG system with configuration."""
        self.config = Config()
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.documents = []
        self.paper_metadata = None
        self._initialization_complete = False
        
        # Initialize components with retries
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components with proper error handling."""
        try:
            # Initialize LLM first
            if not self._initialize_llm():
                return False
            
            # Initialize embeddings
            if not self._initialize_embeddings():
                return False
            
            self._initialization_complete = True
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize RAG system components: {str(e)}")
            return False
    
    def _initialize_llm(self):
        """Initialize the Groq LLM with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not self.config.GROQ_API_KEY:
                    raise ValueError("GROQ_API_KEY not found in environment variables")
                
                self.llm = ChatGroq(
                    groq_api_key=self.config.GROQ_API_KEY,
                    model_name=self.config.LLM_MODEL,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                    timeout=30  # Add timeout
                )
                
                # Test the LLM with a simple query
                test_response = self.llm.invoke("Hello")
                if test_response:
                    display_success("LLM initialized and tested successfully")
                    return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"LLM initialization attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
                else:
                    display_error(f"Failed to initialize Groq LLM after {max_retries} attempts: {str(e)}")
                    return False
        return False
    
    def _initialize_embeddings(self):
        """Initialize the embedding model with fallback options."""
        embedding_models = [
            self.config.EMBEDDING_MODEL,
            "paraphrase-MiniLM-L3-v2",
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2"
        ]
        
        for model_name in embedding_models:
            try:
                # Configure SSL settings for HuggingFace downloads
                os.environ['CURL_CA_BUNDLE'] = ''
                os.environ['REQUESTS_CA_BUNDLE'] = ''
                
                st.info(f"Trying to load embedding model: {model_name}")
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        'device': 'cpu',
                        'trust_remote_code': True
                    },
                    encode_kwargs={
                        'normalize_embeddings': True
                    }
                )
                
                # Test the embeddings
                test_embedding = self.embeddings.embed_query("test")
                if test_embedding and len(test_embedding) > 0:
                    display_success(f"Embedding model '{model_name}' initialized successfully")
                    return True
                
            except Exception as e:
                st.warning(f"Failed to load embedding model '{model_name}': {str(e)}")
                continue
        
        display_error("Failed to initialize any embedding model")
        return False
    
    def process_paper(self, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Process paper text and create vector store using FAISS with comprehensive error handling.
        
        Args:
            text (str): Paper text content
            metadata (Dict[str, Any]): Paper metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Reset any existing state
            self._reset_processing_state()
            
            # Validate initialization
            if not self._initialization_complete:
                st.error("RAG system components not properly initialized")
                return False
            
            # Validate inputs
            if not self._validate_inputs(text, metadata):
                return False
            
            self.paper_metadata = metadata
            
            # Process text into chunks
            if not self._create_document_chunks(text):
                return False
            
            # Create vector store
            if not self._create_vector_store():
                return False
            
            # Set up retriever
            if not self._setup_retriever():
                return False
            
            # Create RAG chain
            if not self._create_rag_chain():
                return False
            
            # Final validation
            if not self.is_ready():
                st.error("RAG system setup completed but final validation failed")
                return False
            
            display_success(f"Successfully processed paper into {len(self.documents)} chunks")
            return True
            
        except Exception as e:
            display_error(f"Failed to process paper: {str(e)}")
            st.error(f"Detailed error: {str(e)}")
            self._reset_processing_state()
            return False
    
    def _reset_processing_state(self):
        """Reset processing state."""
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.documents = []
        self.paper_metadata = None
    
    def _validate_inputs(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Validate inputs for processing."""
        if not text or not text.strip():
            display_error("Paper text is empty or invalid")
            return False
            
        if len(text.strip()) < 100:
            display_error("Paper text is too short (less than 100 characters)")
            return False
            
        if not self.embeddings:
            display_error("Embeddings not initialized")
            return False
            
        if not self.llm:
            display_error("LLM not initialized")
            return False
        
        return True
    
    def _create_document_chunks(self, text: str) -> bool:
        """Create document chunks from text."""
        try:
            st.info("Creating document chunks...")
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            splits = text_splitter.split_text(text)
            if not splits:
                display_error("No text chunks created from paper")
                return False
            
            # Create documents
            self.documents = []
            for i, chunk in enumerate(splits):
                if chunk.strip():  # Only add non-empty chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': self.paper_metadata.get('title', 'Unknown'),
                            'chunk_id': i,
                            'paper_id': self.paper_metadata.get('entry_id', 'unknown'),
                            'authors': ', '.join(self.paper_metadata.get('authors', ['Unknown'])),
                            'categories': ', '.join(self.paper_metadata.get('categories', ['Unknown']))
                        }
                    )
                    self.documents.append(doc)
            
            if not self.documents:
                display_error("No valid documents created from text chunks")
                return False
            
            st.success(f"Created {len(self.documents)} document chunks")
            return True
            
        except Exception as e:
            display_error(f"Failed to create document chunks: {str(e)}")
            return False
    
    def _create_vector_store(self) -> bool:
        """Create FAISS vector store."""
        try:
            st.info("Creating FAISS vector store...")
            
            # Create vector store with batch processing for large documents
            batch_size = 50  # Process in smaller batches
            
            if len(self.documents) <= batch_size:
                # Small document set, process all at once
                self.vectorstore = FAISS.from_documents(
                    documents=self.documents,
                    embedding=self.embeddings
                )
            else:
                # Large document set, process in batches
                st.info(f"Processing {len(self.documents)} documents in batches of {batch_size}")
                
                # Create initial vector store with first batch
                first_batch = self.documents[:batch_size]
                self.vectorstore = FAISS.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings
                )
                
                # Add remaining documents in batches
                for i in range(batch_size, len(self.documents), batch_size):
                    batch = self.documents[i:i + batch_size]
                    batch_vectorstore = FAISS.from_documents(
                        documents=batch,
                        embedding=self.embeddings
                    )
                    self.vectorstore.merge_from(batch_vectorstore)
                    
                    progress = min((i + batch_size) / len(self.documents), 1.0)
                    st.progress(progress)
            
            if not self.vectorstore:
                display_error("Failed to create vector store")
                return False
            
            st.success("Vector store created successfully")
            return True
            
        except Exception as e:
            display_error(f"Failed to create vector store: {str(e)}")
            return False
    
    def _setup_retriever(self) -> bool:
        """Set up document retriever."""
        try:
            st.info("Setting up document retriever...")
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            if not self.retriever:
                display_error("Failed to create retriever")
                return False
            
            # Test the retriever
            test_results = self.retriever.get_relevant_documents("test query")
            if not test_results:
                st.warning("Retriever created but test query returned no results")
            
            st.success("Document retriever set up successfully")
            return True
            
        except Exception as e:
            display_error(f"Failed to set up retriever: {str(e)}")
            return False
    
    def _create_rag_chain(self) -> bool:
        """Create the RAG chain for question answering."""
        try:
            st.info("Creating RAG chain...")
            
            if not self.retriever or not self.llm:
                display_error("Retriever or LLM not available for RAG chain creation")
                return False
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant helping users understand research papers. Use the following context from the paper to answer the question accurately and comprehensively.

**Context from the paper:**
{context}

**Question:** {question}

**Instructions:**
1. Answer based primarily on the provided context from the paper
2. If the context doesn't contain enough information, clearly state what information is missing
3. Provide specific details and explanations when possible
4. Use clear, accessible language while maintaining technical accuracy
5. If relevant, mention specific sections, methodologies, or findings from the paper
6. Don't make up information not present in the context

**Answer:**
""")
            
            # Create the RAG chain
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Test the RAG chain
            try:
                test_answer = self.rag_chain.invoke("What is this paper about?")
                if test_answer and len(test_answer.strip()) > 0:
                    st.success("RAG chain created and tested successfully")
                    return True
                else:
                    st.warning("RAG chain created but test query failed")
                    return False
            except Exception as e:
                st.warning(f"RAG chain created but test failed: {str(e)}")
                return True  # Still return True as chain was created
            
        except Exception as e:
            display_error(f"Failed to create RAG chain: {str(e)}")
            return False
    
    def ask_question(self, question: str) -> Tuple[str, List[Document]]:
        """
        Ask a question about the paper and get an answer with sources.
        
        Args:
            question (str): User question
            
        Returns:
            Tuple[str, List[Document]]: Answer and source documents
        """
        try:
            # Comprehensive validation
            if not question or not question.strip():
                return "Error: Question cannot be empty.", []
            
            if not self.is_ready():
                return "Error: RAG system not properly initialized. Please process a paper first.", []
            
            # Get relevant documents for sources
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return "No relevant content found in the paper for this question. Try rephrasing your question or asking about different aspects of the paper.", []
            
            # Generate answer using the RAG chain
            answer = self.rag_chain.invoke(question)
            
            if not answer or not answer.strip():
                return "Failed to generate an answer. Please try rephrasing your question.", relevant_docs
            
            return answer, relevant_docs
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            st.error(error_msg)
            return error_msg, []
    
    def get_suggested_questions(self) -> List[str]:
        """Generate suggested questions based on the paper content."""
        if not self.paper_metadata:
            return [
                "What is the main contribution of this paper?",
                "What methodology was used in this research?",
                "What were the key findings or results?",
                "What are the limitations of this study?"
            ]
        
        # Base questions for any paper
        questions = [
            "What is the main contribution of this paper?",
            "What methodology was used in this research?",
            "What were the key findings or results?",
            "What are the limitations of this study?",
            "How does this work compare to previous research?",
            "What are the practical applications of this research?",
            "What future work is suggested by the authors?",
            "What datasets were used in this study?"
        ]
        
        # Add domain-specific questions based on categories
        categories = self.paper_metadata.get('categories', [])
        if any('cs.AI' in cat or 'cs.LG' in cat or 'cs.ML' in cat for cat in categories):
            questions.extend([
                "What machine learning algorithms were employed?",
                "How was the model evaluated?",
                "What were the performance metrics achieved?"
            ])
        
        if any('cs.CV' in cat for cat in categories):
            questions.extend([
                "What computer vision techniques were used?",
                "What image datasets were utilized?"
            ])
        
        if any('cs.NLP' in cat or 'cs.CL' in cat for cat in categories):
            questions.extend([
                "What natural language processing methods were applied?",
                "What language models were used?"
            ])
        
        return questions
    
    def get_paper_summary(self) -> str:
        """Generate a comprehensive summary of the paper."""
        try:
            if not self.is_ready():
                return "Error: RAG system not initialized. Please process a paper first."
            
            summary_question = """
            Provide a comprehensive summary of this research paper including:
            1. The main research question or problem addressed
            2. The methodology and approach used
            3. Key findings and results
            4. Main contributions to the field
            5. Limitations and future work suggestions
            
            Make the summary clear and accessible while maintaining technical accuracy.
            """
            
            summary, _ = self.ask_question(summary_question)
            return summary
            
        except Exception as e:
            return f"Unable to generate summary: {str(e)}"
    
    def search_paper_content(self, query: str, k: int = 5) -> List[Document]:
        """Search for specific content in the paper."""
        try:
            if not self.vectorstore:
                display_error("Vector store not initialized")
                return []
            
            if not query or not query.strip():
                display_error("Search query cannot be empty")
                return []
            
            # Use FAISS similarity search
            results = self.vectorstore.similarity_search(query, k=k)
            return results
            
        except Exception as e:
            display_error(f"Search failed: {str(e)}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the current RAG system state."""
        stats = {
            'paper_processed': self.paper_metadata is not None,
            'total_chunks': len(self.documents),
            'embedding_model': self.config.EMBEDDING_MODEL,
            'llm_model': self.config.LLM_MODEL,
            'chunk_size': self.config.CHUNK_SIZE,
            'chunk_overlap': self.config.CHUNK_OVERLAP,
            'vectorstore_initialized': self.vectorstore is not None,
            'retriever_initialized': self.retriever is not None,
            'rag_chain_initialized': self.rag_chain is not None,
            'system_ready': self.is_ready()
        }
        
        if self.paper_metadata:
            stats.update({
                'paper_title': self.paper_metadata.get('title', 'Unknown'),
                'paper_authors': len(self.paper_metadata.get('authors', [])),
                'paper_categories': ', '.join(self.paper_metadata.get('categories', []))
            })
        
        return stats
    
    def reset_system(self):
        """Reset the RAG system to initial state."""
        self._reset_processing_state()
        display_success("RAG system reset successfully")
        
    def is_ready(self) -> bool:
        """Check if the RAG system is ready to answer questions."""
        checks = [
            self._initialization_complete,
            self.llm is not None,
            self.embeddings is not None,
            self.vectorstore is not None,
            self.retriever is not None,
            self.rag_chain is not None,
            len(self.documents) > 0,
            self.paper_metadata is not None
        ]
        
        ready = all(checks)
        
        if not ready:
            # Debug information
            debug_info = {
                'initialization_complete': self._initialization_complete,
                'llm_ready': self.llm is not None,
                'embeddings_ready': self.embeddings is not None,
                'vectorstore_ready': self.vectorstore is not None,
                'retriever_ready': self.retriever is not None,
                'rag_chain_ready': self.rag_chain is not None,
                'has_documents': len(self.documents) > 0,
                'has_metadata': self.paper_metadata is not None
            }
            st.write("RAG System Debug Info:", debug_info)
        
        return ready