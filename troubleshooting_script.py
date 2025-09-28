#!/usr/bin/env python3
"""
Comprehensive troubleshooting script for arXiv Paper Explainer
This will help identify the exact cause of the PDF file issues
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

def test_basic_imports():
    """Test basic imports"""
    print("="*60)
    print("TESTING BASIC IMPORTS")
    print("="*60)
    
    modules_to_test = [
        "streamlit",
        "arxiv", 
        "pypdf",
        "tempfile",
        "os"
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}: OK")
        except ImportError as e:
            print(f"❌ {module}: FAILED - {e}")
            return False
    
    return True

def test_temp_directory_access():
    """Test temporary directory access and file creation"""
    print("\n" + "="*60)
    print("TESTING TEMPORARY DIRECTORY ACCESS")
    print("="*60)
    
    try:
        # Test system temp directory
        system_temp = tempfile.gettempdir()
        print(f"System temp directory: {system_temp}")
        print(f"Temp directory exists: {os.path.exists(system_temp)}")
        print(f"Temp directory writable: {os.access(system_temp, os.W_OK)}")
        
        # Test creating temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.test') as tmp:
            tmp_path = tmp.name
            tmp.write(b"test content")
        
        print(f"Created temp file: {tmp_path}")
        print(f"Temp file exists: {os.path.exists(tmp_path)}")
        
        # Read back
        with open(tmp_path, 'rb') as f:
            content = f.read()
            print(f"Read back content: {content}")
        
        # Cleanup
        os.unlink(tmp_path)
        print("✅ Temporary file operations: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Temporary file operations: FAILED - {e}")
        traceback.print_exc()
        return False

def test_pdf_processing():
    """Test PDF processing with a simple PDF"""
    print("\n" + "="*60)
    print("TESTING PDF PROCESSING")
    print("="*60)
    
    try:
        from pypdf import PdfReader
        
        # Create a simple PDF content for testing
        # We'll create a minimal PDF-like structure
        test_pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000174 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
268
%%EOF"""
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(test_pdf_content)
            pdf_path = tmp.name
        
        print(f"Created test PDF: {pdf_path}")
        print(f"File exists: {os.path.exists(pdf_path)}")
        print(f"File size: {os.path.getsize(pdf_path)} bytes")
        
        # Try to read with PyPDF
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                print(f"PDF pages: {len(reader.pages)}")
                print("✅ PDF processing: OK")
            result = True
        except Exception as pdf_error:
            print(f"❌ PDF processing: FAILED - {pdf_error}")
            result = False
        
        # Cleanup
        os.unlink(pdf_path)
        return result
        
    except Exception as e:
        print(f"❌ PDF processing setup: FAILED - {e}")
        traceback.print_exc()
        return False

def test_arxiv_connection():
    """Test arXiv connection and paper download"""
    print("\n" + "="*60)
    print("TESTING ARXIV CONNECTION")
    print("="*60)
    
    try:
        import arxiv
        
        # Create client
        client = arxiv.Client()
        print("✅ arXiv client created")
        
        # Search for a simple paper
        search = arxiv.Search(id_list=["1706.03762"])  # Attention Is All You Need
        
        paper = None
        for result in client.results(search):
            paper = result
            break
        
        if paper:
            print(f"✅ Found paper: {paper.title[:50]}...")
            
            # Try to download
            temp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(temp_dir, "test_paper.pdf")
            
            try:
                paper.download_pdf(filename=pdf_path)
                
                if os.path.exists(pdf_path):
                    file_size = os.path.getsize(pdf_path)
                    print(f"✅ Downloaded PDF: {file_size} bytes")
                    
                    # Test reading
                    from pypdf import PdfReader
                    with open(pdf_path, 'rb') as f:
                        reader = PdfReader(f)
                        pages = len(reader.pages)
                        print(f"✅ PDF readable: {pages} pages")
                    
                    # Cleanup
                    os.unlink(pdf_path)
                    os.rmdir(temp_dir)
                    
                    return True
                else:
                    print("❌ PDF file not created")
                    return False
                    
            except Exception as download_error:
                print(f"❌ Download failed: {download_error}")
                return False
        else:
            print("❌ Paper not found")
            return False
            
    except Exception as e:
        print(f"❌ arXiv connection: FAILED - {e}")
        traceback.print_exc()
        return False

def test_full_workflow():
    """Test the complete workflow with our classes"""
    print("\n" + "="*60)
    print("TESTING FULL WORKFLOW")
    print("="*60)
    
    try:
        # Import our classes
        sys.path.append('.')
        from paper_processor import PaperProcessor
        
        processor = PaperProcessor()
        print("✅ PaperProcessor created")
        
        # Test with a known paper
        url = "https://arxiv.org/abs/1706.03762"
        print(f"Testing with URL: {url}")
        
        # Download
        metadata = processor.download_arxiv_paper(url)
        
        if metadata:
            print(f"✅ Download successful")
            print(f"   Title: {metadata['title'][:50]}...")
            print(f"   Path: {metadata['local_path']}")
            print(f"   File exists: {os.path.exists(metadata['local_path'])}")
            print(f"   File size: {os.path.getsize(metadata['local_path'])} bytes")
            
            # Extract text
            text = processor.extract_text_from_pdf(metadata['local_path'])
            
            if text:
                print(f"✅ Text extraction successful: {len(text)} characters")
                print(f"   Sample: {text[:100]}...")
                
                # Cleanup
                processor.cleanup_temp_files()
                return True
            else:
                print("❌ Text extraction failed")
                return False
        else:
            print("❌ Download failed")
            return False
            
    except Exception as e:
        print(f"❌ Full workflow: FAILED - {e}")
        traceback.print_exc()
        return False

def test_permissions():
    """Test file system permissions"""
    print("\n" + "="*60)
    print("TESTING FILE SYSTEM PERMISSIONS")
    print("="*60)
    
    try:
        # Test current directory
        current_dir = os.getcwd()
        print(f"Current directory: {current_dir}")
        print(f"Current dir writable: {os.access(current_dir, os.W_OK)}")
        
        # Test creating file in current directory
        test_file = os.path.join(current_dir, "test_permission.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        
        print(f"✅ Can create files in current directory")
        os.unlink(test_file)
        
        # Test temp directory permissions
        temp_dir = tempfile.gettempdir()
        print(f"Temp directory: {temp_dir}")
        print(f"Temp dir exists: {os.path.exists(temp_dir)}")
        print(f"Temp dir writable: {os.access(temp_dir, os.W_OK)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Permission test: FAILED - {e}")
        return False

def main():
    """Run all tests"""
    print("ARXIV PAPER EXPLAINER - COMPREHENSIVE TROUBLESHOOTING")
    print("This will help identify the exact cause of your issues")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Temporary Directory Access", test_temp_directory_access),
        ("PDF Processing", test_pdf_processing),
        ("File System Permissions", test_permissions),
        ("arXiv Connection", test_arxiv_connection),
        ("Full Workflow", test_full_workflow)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system should work correctly.")
        print("If you're still having issues, the problem might be in the Streamlit app itself.")
    else:
        print("\n⚠️ Some tests failed. This explains why your app isn't working.")
        print("\nRecommended fixes:")
        
        if not results.get("Basic Imports", True):
            print("- Install missing packages: pip install -r requirements.txt")
        
        if not results.get("Temporary Directory Access", True):
            print("- Check disk space and permissions")
            print("- Try running as administrator/sudo")
        
        if not results.get("PDF Processing", True):
            print("- Try: pip install --upgrade pypdf")
        
        if not results.get("File System Permissions", True):
            print("- Check file system permissions")
            print("- Try running from a different directory")
        
        if not results.get("arXiv Connection", True):
            print("- Check internet connection")
            print("- Check if behind corporate firewall")
        
        if not results.get("Full Workflow", True):
            print("- Check the paper_processor.py file")
            print("- Make sure all dependencies are installed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)