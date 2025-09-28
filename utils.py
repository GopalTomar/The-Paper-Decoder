"""
Utility functions for the arXiv Paper Explainer application.
"""

import re
import streamlit as st
from typing import Optional, Tuple
import requests
from urllib.parse import urlparse
import tempfile
import os

def is_valid_url(url: str) -> bool:
    """
    Simple URL validation function to replace validators dependency.
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if valid URL format
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def validate_arxiv_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if the provided URL is a valid arXiv URL.
    
    Args:
        url (str): URL to validate
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, paper_id)
    """
    if not url:
        return False, None
    
    # Clean the URL
    url = url.strip()
    
    # Check if it's a valid URL format
    if not is_valid_url(url):
        return False, None
    
    # arXiv URL patterns
    patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?',
        r'export\.arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return True, match.group(1)
    
    return False, None

def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    """
    Extract arXiv paper ID from URL.
    
    Args:
        url (str): arXiv URL
        
    Returns:
        Optional[str]: Paper ID if found, None otherwise
    """
    is_valid, paper_id = validate_arxiv_url(url)
    return paper_id if is_valid else None

def format_paper_metadata(paper) -> dict:
    """
    Format paper metadata for display.
    
    Args:
        paper: arXiv paper object
        
    Returns:
        dict: Formatted metadata
    """
    return {
        'title': paper.title,
        'authors': [author.name for author in paper.authors],
        'summary': paper.summary,
        'published': paper.published.strftime('%Y-%m-%d'),
        'categories': paper.categories,
        'pdf_url': paper.pdf_url,
        'entry_id': paper.entry_id
    }

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and formatting.
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove references to figures/tables that might be out of context
    text = re.sub(r'\(see\s+(?:Figure|Table|Fig\.)\s+\d+\)', '', text, flags=re.IGNORECASE)
    
    # Clean up line breaks
    text = re.sub(r'\n+', '\n', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'\.{3,}', '...', text)  # Multiple dots
    text = re.sub(r'-{2,}', '--', text)    # Multiple dashes
    
    return text.strip()

def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary location.
    
    Args:
        uploaded_file: Streamlit uploaded file
        
    Returns:
        str: Path to saved file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def create_download_link(text: str, filename: str) -> str:
    """
    Create a download link for text content.
    
    Args:
        text (str): Text content
        filename (str): Filename for download
        
    Returns:
        str: HTML download link
    """
    import base64
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def display_error(error_message: str, error_type: str = "error"):
    """
    Display styled error message.
    
    Args:
        error_message (str): Error message to display
        error_type (str): Type of error (error, warning, info)
    """
    if error_type == "error":
        st.error(f"❌ {error_message}")
    elif error_type == "warning":
        st.warning(f"⚠️ {error_message}")
    elif error_type == "info":
        st.info(f"ℹ️ {error_message}")
    else:
        st.error(f"❌ {error_message}")

def display_success(message: str):
    """
    Display success message.
    
    Args:
        message (str): Success message to display
    """
    st.success(f"✅ {message}")

def setup_page_config():
    """
    Set up Streamlit page configuration.
    """
    st.set_page_config(
        page_title="arXiv Paper Explainer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def add_custom_css():
    """
    Add custom CSS styling to the application.
    """
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #FF6B6B;
        background-color: rgba(255, 107, 107, 0.1);
    }
    
    .paper-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50, #3498DB);
    }
    
    .uploadedFile {
        border: 2px dashed #FF6B6B;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: rgba(255, 107, 107, 0.05);
    }
    
    .stAlert > div {
        background-color: rgba(255, 107, 107, 0.1);
        border: 1px solid #FF6B6B;
        border-radius: 10px;
    }
    
    .stSuccess > div {
        background-color: rgba(76, 205, 196, 0.1);
        border: 1px solid #4ECDC4;
        border-radius: 10px;
    }
    
    .element-container:has(.stAlert) {
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def format_response_with_sources(response: str, sources: list) -> str:
    """
    Format response with source citations.
    
    Args:
        response (str): Generated response
        sources (list): List of source documents
        
    Returns:
        str: Formatted response with sources
    """
    formatted_response = f"{response}\n\n"
    
    if sources:
        formatted_response += "**📚 Sources:**\n"
        for i, source in enumerate(sources[:3], 1):  # Limit to top 3 sources
            content_preview = truncate_text(source.page_content, 150)
            formatted_response += f"{i}. {content_preview}...\n"
    
    return formatted_response

def check_network_connectivity() -> bool:
    """
    Check if network connectivity is available.
    
    Returns:
        bool: True if connected, False otherwise
    """
    try:
        # Try to connect to a reliable service
        response = requests.get('https://httpbin.org/status/200', timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_info() -> dict:
    """
    Get system information for debugging.
    
    Returns:
        dict: System information
    """
    import platform
    import sys
    
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'streamlit_version': st.__version__ if hasattr(st, '__version__') else 'unknown',
        'working_directory': os.getcwd(),
        'temp_directory': tempfile.gettempdir()
    }

def validate_groq_api_key(api_key: str) -> bool:
    """
    Validate Groq API key format.
    
    Args:
        api_key (str): API key to validate
        
    Returns:
        bool: True if valid format
    """
    if not api_key or api_key == 'your_groq_api_key_here':
        return False
    
    # Basic format check - Groq keys typically start with 'gsk_'
    if api_key.startswith('gsk_') and len(api_key) > 20:
        return True
    
    return False