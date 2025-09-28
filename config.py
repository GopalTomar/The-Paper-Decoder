"""
Configuration settings for the arXiv Paper Explainer application.
"""

import os
import ssl
import urllib3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure SSL settings for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

class Config:
    """Application configuration class."""
    
    # API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    
    # Application Settings
    APP_TITLE = os.getenv('APP_TITLE', 'arXiv Paper Explainer & Q&A System')
    APP_DESCRIPTION = os.getenv('APP_DESCRIPTION', 'AI-powered research paper analysis and Q&A system using RAG')
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 50))
    
    # Vector Database
    CHROMA_PERSIST_DIRECTORY = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    
    # Text Processing
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
    
    # LLM Configuration
    LLM_MODEL = os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.1))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 1000))
    
    # Streamlit Configuration
    PAGE_TITLE = "arXiv Paper Explainer"
    PAGE_ICON = "📬"
    LAYOUT = "wide"
    
    # UI Colors and Styling
    PRIMARY_COLOR = "#FF6B6B"
    BACKGROUND_COLOR = "#0E1117"
    SECONDARY_BACKGROUND_COLOR = "#262730"
    TEXT_COLOR = "#FAFAFA"
    
    # Network Configuration for Corporate Environments
    DISABLE_SSL_VERIFY = os.getenv('DISABLE_SSL_VERIFY', 'true').lower() == 'true'
    HTTP_PROXY = os.getenv('HTTP_PROXY', '')
    HTTPS_PROXY = os.getenv('HTTPS_PROXY', '')
    
    @classmethod
    def setup_network_config(cls):
        """Setup network configuration for corporate environments."""
        if cls.DISABLE_SSL_VERIFY:
            # Disable SSL verification
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Set up proxies if provided
        if cls.HTTP_PROXY or cls.HTTPS_PROXY:
            proxy_dict = {}
            if cls.HTTP_PROXY:
                proxy_dict['http'] = cls.HTTP_PROXY
            if cls.HTTPS_PROXY:
                proxy_dict['https'] = cls.HTTPS_PROXY
            
            # Set environment variables for requests
            for key, value in proxy_dict.items():
                os.environ[f'{key}_proxy'] = value
                os.environ[f'{key.upper()}_PROXY'] = value
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration values."""
        errors = []
        
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required. Please set it in your .env file.")
        
        if not cls.GROQ_API_KEY or cls.GROQ_API_KEY == 'your_groq_api_key_here':
            errors.append("Please update GROQ_API_KEY in your .env file with your actual API key.")
        
        # Validate numeric settings
        try:
            assert cls.MAX_FILE_SIZE_MB > 0
            assert cls.CHUNK_SIZE > 0
            assert cls.CHUNK_OVERLAP >= 0
            assert 0 <= cls.TEMPERATURE <= 2
            assert cls.MAX_TOKENS > 0
        except AssertionError:
            errors.append("Invalid numeric configuration values.")
        
        if errors:
            raise ValueError('\n'.join(errors))
        
        # Setup network configuration
        cls.setup_network_config()
        
        return True
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary for easy access."""
        return {
            'groq_api_key': cls.GROQ_API_KEY,
            'app_title': cls.APP_TITLE,
            'app_description': cls.APP_DESCRIPTION,
            'max_file_size_mb': cls.MAX_FILE_SIZE_MB,
            'chroma_persist_directory': cls.CHROMA_PERSIST_DIRECTORY,
            'embedding_model': cls.EMBEDDING_MODEL,
            'chunk_size': cls.CHUNK_SIZE,
            'chunk_overlap': cls.CHUNK_OVERLAP,
            'llm_model': cls.LLM_MODEL,
            'temperature': cls.TEMPERATURE,
            'max_tokens': cls.MAX_TOKENS,
            'disable_ssl_verify': cls.DISABLE_SSL_VERIFY
        }
    
    @classmethod
    def create_env_template(cls):
        """Create .env template file."""
        template_content = """# arXiv Paper Explainer Configuration

# Required: Get your API key from https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# Optional: HuggingFace API key (for some models)
HUGGINGFACE_API_KEY=

# Application Settings
APP_TITLE=arXiv Paper Explainer & Q&A System
APP_DESCRIPTION=AI-powered research paper analysis and Q&A system using RAG
MAX_FILE_SIZE_MB=50

# Vector Database Settings
CHROMA_PERSIST_DIRECTORY=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Text Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# LLM Configuration
LLM_MODEL=llama-3.1-8b-instant
TEMPERATURE=0.1
MAX_TOKENS=1000

# Network Configuration (for corporate environments)
DISABLE_SSL_VERIFY=true
HTTP_PROXY=
HTTPS_PROXY=
"""
        
        if not os.path.exists('.env'):
            with open('.env', 'w') as f:
                f.write(template_content)
            return True
        return False