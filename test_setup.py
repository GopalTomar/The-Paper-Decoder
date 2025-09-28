#!/usr/bin/env python3
"""
Simple test script to verify arXiv Paper Explainer setup
"""

import os
import sys
from pathlib import Path

def test_python_version():
    """Test Python version compatibility."""
    print("Checking Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires 3.8+")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("\nTesting imports...")
    
    required_modules = [
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
        ("langchain_groq", "LangChain Groq"),
        ("faiss", "FAISS"),
        ("sentence_transformers", "Sentence Transformers"),
        ("arxiv", "arXiv API"),
        ("pypdf", "PyPDF"),
        ("requests", "Requests"),
        ("dotenv", "Python Dotenv"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("plotly", "Plotly")
    ]
    
    failed_imports = []
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All imports successful")
    return True

def test_env_file():
    """Test .env file configuration."""
    print("\nChecking .env file...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Create .env file with your GROQ_API_KEY")
        return False
    
    print("✅ .env file exists")
    return True

def test_api_key():
    """Test API key configuration."""
    print("\nChecking API key...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("❌ GROQ_API_KEY not found in .env file")
            return False
        
        if api_key == 'your_actual_groq_api_key_here':
            print("❌ GROQ_API_KEY not set (still has placeholder value)")
            return False
        
        if not api_key.startswith('gsk_'):
            print("⚠️ GROQ_API_KEY format might be incorrect (should start with 'gsk_')")
            return False
        
        print("✅ API key configured")
        return True
        
    except Exception as e:
        print(f"❌ Error checking API key: {e}")
        return False

def test_groq_connection():
    """Test connection to Groq API."""
    print("\nTesting Groq API connection...")
    
    try:
        from langchain_groq import ChatGroq
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key or api_key == 'your_actual_groq_api_key_here':
            print("❌ Cannot test - API key not configured")
            return False
        
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=50,
            timeout=10
        )
        
        response = llm.invoke("Hello, respond with 'API test successful'")
        
        if response and hasattr(response, 'content'):
            print("✅ Groq API connection successful")
            return True
        else:
            print("❌ Groq API test failed - no response")
            return False
            
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False

def test_embedding_model():
    """Test embedding model initialization."""
    print("\nTesting embedding model...")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test embedding
        test_embedding = embeddings.embed_query("test")
        
        if test_embedding and len(test_embedding) > 0:
            print("✅ Embedding model working")
            return True
        else:
            print("❌ Embedding model test failed")
            return False
            
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

def test_basic_components():
    """Test basic application components."""
    print("\nTesting application components...")
    
    try:
        # Test config
        from config import Config
        config = Config()
        config.validate_config()
        print("✅ Configuration")
        
        # Test paper processor
        from paper_processor import PaperProcessor
        processor = PaperProcessor()
        print("✅ Paper processor")
        
        # Test RAG system
        from rag_system import RAGSystem
        rag = RAGSystem()
        print("✅ RAG system")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("arXiv Paper Explainer - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Required Imports", test_imports),
        ("Environment File", test_env_file),
        ("API Key", test_api_key),
        ("Groq Connection", test_groq_connection),
        ("Embedding Model", test_embedding_model),
        ("Application Components", test_basic_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nTo start the application:")
        print("  streamlit run app.py")
        print("\nThen open: http://localhost:8501")
    else:
        print("⚠️ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("  1. Install requirements: pip install -r requirements.txt")
        print("  2. Set up .env file with your GROQ_API_KEY")
        print("  3. Check internet connection for model downloads")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)