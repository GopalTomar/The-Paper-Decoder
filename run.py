"""
Application runner script for the arXiv Paper Explainer.
This script provides a simple way to start the Streamlit application with proper error handling.
"""

import os
import sys
import subprocess
import argparse
import ssl
import urllib3
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == "win32":
    # Set UTF-8 encoding for stdout
    import codecs
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
        # Set console to UTF-8 mode
        os.system("chcp 65001 >nul 2>&1")
    except:
        # Fallback if encoding setup fails
        pass

# Disable SSL warnings for corporate environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

def print_safe(text, fallback_text=None):
    """Print text with encoding fallback."""
    try:
        print(text)
    except UnicodeEncodeError:
        if fallback_text:
            print(fallback_text)
        else:
            # Remove Unicode characters
            safe_text = text.encode('ascii', 'ignore').decode('ascii')
            print(safe_text)

def print_banner():
    """Print application banner."""
    banner_unicode = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║            📚 arXiv Paper Explainer & Q&A System            ║
    ║                                                              ║
    ║           AI-powered research paper analysis using RAG       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    banner_fallback = """
    ==============================================================
    
                arXiv Paper Explainer & Q&A System
    
           AI-powered research paper analysis using RAG
    
    ==============================================================
    """
    
    print_safe(banner_unicode, banner_fallback)

def check_requirements():
    """Check if all required packages are installed."""
    print_safe("🔍 Checking required packages...", "Checking required packages...")
    
    required_packages = [
        'streamlit',
        'langchain', 
        'langchain_groq',
        'chromadb',
        'arxiv',
        'pypdf',
        'sentence_transformers',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print_safe(f"   ✅ {package}", f"   OK {package}")
        except ImportError:
            print_safe(f"   ❌ {package}", f"   MISSING {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print_safe(f"\n❌ Missing packages: {', '.join(missing_packages)}", 
                  f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print_safe("✅ All required packages are installed", "All required packages are installed")
    return True

def check_env_file():
    """Check if .env file exists and has required variables."""
    print_safe("\n🔍 Checking configuration...", "\nChecking configuration...")
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print_safe("❌ .env file not found", ".env file not found")
        print("Creating .env file from template...")
        
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
HTTP_PROXY=
HTTPS_PROXY=
"""
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print_safe("✅ Created .env file", "Created .env file")
        print_safe("⚠️  Please edit .env file and add your GROQ_API_KEY", 
                  "WARNING: Please edit .env file and add your GROQ_API_KEY")
        return False
    
    # Check if GROQ_API_KEY is set
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        groq_key = os.getenv('GROQ_API_KEY')
        if not groq_key or groq_key == 'your_groq_api_key_here':
            print_safe("❌ GROQ_API_KEY not properly configured in .env file", 
                      "GROQ_API_KEY not properly configured in .env file")
            print("Please edit .env file and add your API key from: https://console.groq.com/keys")
            return False
        
        print_safe("✅ Configuration looks good", "Configuration looks good")
        return True
        
    except ImportError:
        print_safe("❌ python-dotenv not installed", "python-dotenv not installed")
        return False

def test_basic_functionality():
    """Test basic app functionality."""
    print_safe("\n🧪 Testing basic functionality...", "\nTesting basic functionality...")
    
    try:
        # Test config loading
        sys.path.append('.')
        from config import Config
        config = Config()
        config.validate_config()
        print_safe("   ✅ Configuration validation", "   Configuration validation OK")
        
        # Test component initialization
        from paper_processor import PaperProcessor
        processor = PaperProcessor()
        print_safe("   ✅ Paper processor initialization", "   Paper processor initialization OK")
        
        from rag_system import RAGSystem
        rag = RAGSystem()
        print_safe("   ✅ RAG system initialization", "   RAG system initialization OK")
        
        print_safe("✅ Basic functionality test passed", "Basic functionality test passed")
        return True
        
    except Exception as e:
        error_msg = f"Functionality test failed: {str(e)}"
        print_safe(f"❌ {error_msg}", error_msg)
        print("Check the detailed error above")
        return False

def run_streamlit(port=8501, host="localhost", debug=False):
    """Run the Streamlit application."""
    app_file = Path("app.py")
    
    if not app_file.exists():
        print_safe("❌ app.py not found in current directory", "app.py not found in current directory")
        return False
    
    print_safe(f"\n🚀 Starting arXiv Paper Explainer...", f"\nStarting arXiv Paper Explainer...")
    print(f"   URL: http://{host}:{port}")
    print("   Press Ctrl+C to stop the application")
    print("   Check this console for detailed error messages")
    print("-" * 50)
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        if debug:
            cmd.extend(["--logger.level", "debug"])
        
        # Set environment variables for SSL handling
        env = os.environ.copy()
        env['PYTHONHTTPSVERIFY'] = '0'
        env['CURL_CA_BUNDLE'] = ''
        env['REQUESTS_CA_BUNDLE'] = ''
        
        subprocess.run(cmd, check=True, env=env)
        
    except KeyboardInterrupt:
        print_safe("\n👋 Application stopped by user", "\nApplication stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Streamlit: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure port", port, "is not in use")
        print("2. Try a different port: python run.py --port 8502")
        print("3. Check if streamlit is installed: pip install streamlit")
        return False
    except FileNotFoundError:
        print("Streamlit not found. Please install it:")
        print("   pip install streamlit")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def install_requirements():
    """Install required packages from requirements.txt."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print_safe("❌ requirements.txt not found", "requirements.txt not found")
        return False
    
    print_safe("📦 Installing required packages...", "Installing required packages...")
    print("This may take a few minutes...")
    
    try:
        # Upgrade pip first
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True, capture_output=True)
        
        # Install requirements with corporate network support
        cmd = [
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file),
            "--trusted-host", "pypi.org",
            "--trusted-host", "pypi.python.org", 
            "--trusted-host", "files.pythonhosted.org"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print_safe("✅ All packages installed successfully", "All packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])  # Last 500 chars
        if e.stderr:
            print("STDERR:", e.stderr[-500:])  # Last 500 chars
        print("\nTry manual installation:")
        print("pip install -r requirements.txt")
        return False

def setup_project():
    """Set up the project from scratch."""
    print_banner()
    print_safe("🛠️  Setting up arXiv Paper Explainer project...", "Setting up arXiv Paper Explainer project...")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check environment
    check_env_file()
    
    print_safe("\n🎉 Project setup complete!", "\nProject setup complete!")
    print_safe("\n📋 Next steps:", "\nNext steps:")
    print("1. Edit .env file and add your GROQ_API_KEY")
    print("   Get your key from: https://console.groq.com/keys")
    print("2. Run: python run.py")
    print_safe("\n💡 For troubleshooting, see TROUBLESHOOTING.md", "\nFor troubleshooting, see TROUBLESHOOTING.md")
    
    return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="arXiv Paper Explainer - AI-powered research paper Q&A system"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run the Streamlit app on (default: 8501)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to run the Streamlit app on (default: localhost)"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up the project (install requirements, create .env)"
    )
    
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install/reinstall requirements"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if all requirements and configuration are ready"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run with debug logging"
    )
    
    args = parser.parse_args()
    
    # Handle different commands
    if args.setup:
        setup_project()
        return
    
    if args.install:
        install_requirements()
        return
    
    if args.check:
        print_banner()
        print_safe("🔍 Checking system status...\n", "Checking system status...\n")
        
        checks = [
            ("Requirements", check_requirements),
            ("Configuration", check_env_file),
            ("Functionality", test_basic_functionality)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            if not check_func():
                all_passed = False
        
        print("\n" + "="*50)
        if all_passed:
            print_safe("✅ All checks passed! System ready to run.", "All checks passed! System ready to run.")
            print("Execute: python run.py")
        else:
            print_safe("❌ Some checks failed. Please fix the issues above.", "Some checks failed. Please fix the issues above.")
            print("For help, see TROUBLESHOOTING.md")
        return
    
    # Default: run the application
    print_banner()
    print_safe("🔍 Pre-flight checks...\n", "Pre-flight checks...\n")
    
    # Quick checks before starting
    if not check_requirements():
        print_safe("\n💡 Run 'python run.py --install' to install requirements", 
                  "\nRun 'python run.py --install' to install requirements")
        return
    
    if not check_env_file():
        print_safe("\n💡 Run 'python run.py --setup' to configure the project", 
                  "\nRun 'python run.py --setup' to configure the project")
        return
    
    print_safe("✅ System ready!\n", "System ready!\n")
    
    # Start the application
    run_streamlit(port=args.port, host=args.host, debug=args.debug)

if __name__ == "__main__":
    try:
        main()
    except UnicodeEncodeError:
        print("Setup completed successfully despite encoding warnings.")
        print("You can now run the application.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)