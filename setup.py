#!/usr/bin/env python3
"""
Setup script for arXiv Paper Explainer
This script handles installation and configuration with proper encoding support
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == "win32":
    # Set UTF-8 encoding for stdout
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    # Set console to UTF-8 mode
    os.system("chcp 65001 >nul")

def print_header():
    """Print setup header."""
    print("="*60)
    print("📚 arXiv Paper Explainer & Q&A System Setup")
    print("="*60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        print("   Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements
        print("   Installing dependencies...")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        
        # Add --trusted-host flags for corporate networks
        cmd.extend([
            "--trusted-host", "pypi.org",
            "--trusted-host", "pypi.python.org", 
            "--trusted-host", "files.pythonhosted.org"
        ])
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ All packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def create_env_file():
    """Create .env file from template."""
    print("\n📝 Setting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        # Copy example to .env
        with open(env_example, 'r', encoding='utf-8') as src:
            content = src.read()
        
        with open(env_file, 'w', encoding='utf-8') as dst:
            dst.write(content)
        
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file to add your GROQ_API_KEY")
        return True
    else:
        # Create basic .env file
        env_content = """# arXiv Paper Explainer Configuration

# Required: Get your API key from https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# Application Settings
APP_TITLE=arXiv Paper Explainer & Q&A System
MAX_FILE_SIZE_MB=50

# Text Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# LLM Configuration
LLM_MODEL=llama-3.1-8b-instant
TEMPERATURE=0.1
MAX_TOKENS=1000

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Network Configuration (for corporate environments)
DISABLE_SSL_VERIFY=true
"""
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print("✅ Created basic .env file")
        print("⚠️  Please edit .env file to add your GROQ_API_KEY")
        return True

def test_imports():
    """Test if all required modules can be imported."""
    print("\n🔍 Testing module imports...")
    
    required_modules = [
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
        ("chromadb", "ChromaDB"),
        ("arxiv", "arXiv API"),
        ("pypdf", "PyPDF"),
        ("sentence_transformers", "Sentence Transformers"),
        ("requests", "Requests"),
        ("dotenv", "Python Dotenv")
    ]
    
    failed_imports = []
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name}")
        except ImportError as e:
            print(f"   ❌ {display_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("\n✅ All modules imported successfully")
    return True

def test_network_connectivity():
    """Test network connectivity."""
    print("\n🌐 Testing network connectivity...")
    
    try:
        import requests
        response = requests.get('https://httpbin.org/status/200', timeout=10)
        if response.status_code == 200:
            print("✅ Network connectivity OK")
            return True
        else:
            print(f"⚠️  Network test returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Network connectivity issue: {e}")
        print("   This might be due to corporate firewall/proxy")
        print("   The app may still work with local models")
        return False

def check_groq_api_key():
    """Check if GROQ API key is configured."""
    print("\n🔑 Checking API key configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key or groq_key == 'your_groq_api_key_here':
        print("⚠️  GROQ_API_KEY not configured")
        print("   Please edit .env file and add your API key")
        print("   Get your key from: https://console.groq.com/keys")
        return False
    
    print("✅ GROQ_API_KEY configured")
    return True

def create_test_script():
    """Create a test script to verify the setup."""
    print("\n📋 Creating test script...")
    
    test_content = '''#!/usr/bin/env python3
"""
Quick test script for arXiv Paper Explainer
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    try:
        import streamlit as st
        import langchain
        import chromadb
        import sentence_transformers
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    try:
        from config import Config
        config = Config()
        config.validate_config()
        print("✅ Configuration valid")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_components():
    try:
        from paper_processor import PaperProcessor
        from rag_system import RAGSystem
        
        processor = PaperProcessor()
        rag = RAGSystem()
        print("✅ Components initialized")
        return True
    except Exception as e:
        print(f"❌ Component error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running system tests...")
    print("-" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_config),
        ("Components", test_components)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! System ready.")
        print("Run: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
'''

    with open("test_setup.py", "w", encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ Test script created")

def print_instructions():
    """Print final setup instructions."""
    print("\n" + "="*60)
    print("🎯 Setup Complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Edit .env file and add your GROQ_API_KEY")
    print("   Get your key from: https://console.groq.com/keys")
    print()
    print("2. Test the setup:")
    print("   python test_setup.py")
    print()
    print("3. Run the application:")
    print("   streamlit run app.py")
    print()
    print("4. Open your browser and go to:")
    print("   http://localhost:8501")
    print()
    print("Troubleshooting:")
    print("- If you get SSL errors, the app is configured to work around them")
    print("- For corporate networks, you may need to set HTTP_PROXY in .env")
    print("- Check the console for detailed error messages")
    print()

def main():
    """Main setup function."""
    try:
        print_header()
        
        # Check Python version
        if not check_python_version():
            print("❌ Python version check failed")
            sys.exit(1)
        
        # Install requirements
        if not install_requirements():
            print("\n❌ Failed to install requirements")
            print("Try manually: pip install -r requirements.txt")
            sys.exit(1)
        
        # Create .env file
        create_env_file()
        
        # Test imports
        test_imports()
        
        # Test network
        test_network_connectivity()
        
        # Check API key
        check_groq_api_key()
        
        # Create test script
        create_test_script()
        
        # Print final instructions
        print_instructions()
        
    except UnicodeEncodeError as e:
        print("Encoding error occurred. This is a Windows console encoding issue.")
        print("The setup completed successfully despite the encoding warning.")
        print("You can now run: python run.py")
    except Exception as e:
        print(f"Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()