"""
Paper processing module for downloading and extracting text from arXiv papers and uploaded PDFs.
FINAL VERSION - Fixed all temporary file and PDF processing issues.
"""

import arxiv
import tempfile
import os
import ssl
import urllib3
from typing import Optional, Dict, Any, List
import streamlit as st
from pypdf import PdfReader
import requests
from io import BytesIO
import time
import shutil

# Disable SSL warnings for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

from utils import (
    validate_arxiv_url, 
    extract_arxiv_id_from_url, 
    format_paper_metadata,
    clean_text,
    display_error,
    display_success
)
from config import Config

class PaperProcessor:
    """Handle paper downloading, processing, and text extraction."""
    
    def __init__(self):
        self.temp_files = []  # Keep track of temp files for cleanup
        
        # Create a dedicated temp directory for this session
        self.temp_dir = tempfile.mkdtemp(prefix="arxiv_papers_")
        
        # Configure SSL context for arxiv downloads
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    def download_arxiv_paper(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Download paper from arXiv URL with robust error handling.
        """
        try:
            # Validate URL and extract paper ID
            is_valid, paper_id = validate_arxiv_url(url)
            if not is_valid:
                st.error("Invalid arXiv URL. Please provide a valid arXiv paper URL.")
                return None
            
            st.info(f"Searching for paper ID: {paper_id}")
            
            # Create arXiv client and search for paper
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id])
            
            paper = None
            try:
                results = list(client.results(search))
                if results:
                    paper = results[0]
                else:
                    st.error(f"Paper with ID {paper_id} not found on arXiv.")
                    return None
            except Exception as e:
                st.error(f"Error searching arXiv: {str(e)}")
                return None
            
            st.success(f"Found paper: {paper.title}")
            
            # Create a specific filename for this paper
            safe_title = "".join(c for c in paper.title[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            pdf_filename = f"{paper_id}_{safe_title}.pdf"
            pdf_path = os.path.join(self.temp_dir, pdf_filename)
            
            # Download PDF
            with st.spinner("Downloading PDF..."):
                try:
                    # Download to our specific path
                    paper.download_pdf(filename=pdf_path)
                    
                    # Verify download
                    if not os.path.exists(pdf_path):
                        raise Exception("PDF download failed - file not created")
                    
                    file_size = os.path.getsize(pdf_path)
                    if file_size == 0:
                        raise Exception("PDF download failed - file is empty")
                    
                    # Test if it's a valid PDF
                    try:
                        with open(pdf_path, 'rb') as test_file:
                            test_reader = PdfReader(test_file)
                            if len(test_reader.pages) == 0:
                                raise Exception("PDF has no readable pages")
                    except Exception as pdf_error:
                        raise Exception(f"Invalid PDF: {str(pdf_error)}")
                    
                    # Add to our tracking list
                    self.temp_files.append(pdf_path)
                    
                    st.success(f"PDF downloaded: {file_size} bytes")
                    
                except Exception as e:
                    st.error(f"Failed to download PDF: {str(e)}")
                    # Clean up failed download
                    if os.path.exists(pdf_path):
                        try:
                            os.remove(pdf_path)
                        except:
                            pass
                    return None
            
            # Format metadata
            metadata = format_paper_metadata(paper)
            metadata['local_path'] = pdf_path
            metadata['file_exists'] = os.path.exists(pdf_path)
            metadata['file_size'] = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
            
            return metadata
            
        except Exception as e:
            st.error(f"Failed to download arXiv paper: {str(e)}")
            return None
    
    def process_uploaded_pdf(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """
        Process uploaded PDF file with robust handling.
        """
        try:
            # Check file size
            file_content = uploaded_file.getvalue()
            file_size_mb = len(file_content) / (1024 * 1024)
            
            if file_size_mb > Config.MAX_FILE_SIZE_MB:
                st.error(f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({Config.MAX_FILE_SIZE_MB}MB)")
                return None
            
            # Check file type
            if uploaded_file.type != 'application/pdf':
                st.error("Please upload a PDF file.")
                return None
            
            # Create specific filename for uploaded file
            safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in ('.', '-', '_'))
            pdf_path = os.path.join(self.temp_dir, f"uploaded_{safe_filename}")
            
            # Save file
            with st.spinner("Processing uploaded file..."):
                try:
                    # Write file content to our path
                    with open(pdf_path, 'wb') as f:
                        f.write(file_content)
                    
                    # Verify file was written
                    if not os.path.exists(pdf_path):
                        raise Exception("Failed to save uploaded file")
                    
                    saved_size = os.path.getsize(pdf_path)
                    if saved_size == 0:
                        raise Exception("Saved file is empty")
                    
                    # Test PDF validity
                    try:
                        with open(pdf_path, 'rb') as test_file:
                            test_reader = PdfReader(test_file)
                            if len(test_reader.pages) == 0:
                                raise Exception("PDF has no readable pages")
                    except Exception as pdf_error:
                        raise Exception(f"Invalid PDF: {str(pdf_error)}")
                    
                    # Add to tracking
                    self.temp_files.append(pdf_path)
                    
                    st.success(f"File processed: {saved_size} bytes")
                    
                except Exception as e:
                    st.error(f"Failed to process uploaded file: {str(e)}")
                    if os.path.exists(pdf_path):
                        try:
                            os.remove(pdf_path)
                        except:
                            pass
                    return None
            
            # Create metadata
            metadata = {
                'title': uploaded_file.name.replace('.pdf', ''),
                'authors': ['Unknown'],
                'summary': 'User uploaded PDF',
                'published': 'Unknown',
                'categories': ['User Upload'],
                'pdf_url': None,
                'entry_id': f'uploaded_{uploaded_file.name}',
                'local_path': pdf_path,
                'file_exists': os.path.exists(pdf_path),
                'file_size': f"{file_size_mb:.1f}MB"
            }
            
            return metadata
            
        except Exception as e:
            st.error(f"Failed to process uploaded PDF: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text content from PDF file with comprehensive error handling.
        """
        try:
            # Comprehensive path validation
            if not pdf_path:
                st.error("No PDF path provided")
                return None
            
            if not os.path.exists(pdf_path):
                st.error(f"PDF file not found at: {pdf_path}")
                # Try to find the file in our temp directory
                if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                    files = os.listdir(self.temp_dir)
                    st.info(f"Available files in temp directory: {files}")
                return None
            
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                st.error("PDF file is empty")
                return None
            
            st.info(f"Extracting text from: {os.path.basename(pdf_path)} ({file_size} bytes)")
            
            # Extract text with progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Opening PDF file...")
                progress_bar.progress(0.1)
                
                with open(pdf_path, 'rb') as file:
                    reader = PdfReader(file)
                    
                    total_pages = len(reader.pages)
                    if total_pages == 0:
                        st.error("PDF has no pages")
                        return None
                    
                    status_text.text(f"Processing {total_pages} pages...")
                    progress_bar.progress(0.2)
                    
                    # Extract text from all pages
                    text_content = []
                    failed_pages = 0
                    
                    for page_num, page in enumerate(reader.pages):
                        try:
                            # Update progress
                            progress = 0.2 + (0.7 * (page_num + 1) / total_pages)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing page {page_num + 1}/{total_pages}...")
                            
                            text = page.extract_text()
                            if text and text.strip():
                                text_content.append(text)
                            else:
                                failed_pages += 1
                                
                        except Exception as page_error:
                            st.warning(f"Failed to extract text from page {page_num + 1}: {str(page_error)}")
                            failed_pages += 1
                            continue
                    
                    progress_bar.progress(0.9)
                    status_text.text("Cleaning and validating text...")
                    
                    if not text_content:
                        st.error("No readable text found in PDF. The file might be image-based or corrupted.")
                        return None
                    
                    if failed_pages > 0:
                        st.warning(f"Could not extract text from {failed_pages} out of {total_pages} pages")
                    
                    # Join and clean text
                    full_text = '\n\n'.join(text_content)
                    cleaned_text = clean_text(full_text)
                    
                    # Final validation
                    if len(cleaned_text.strip()) < 100:
                        st.error("Extracted text is too short. The PDF might not contain sufficient readable content.")
                        return None
                    
                    progress_bar.progress(1.0)
                    status_text.text("Text extraction complete!")
                    
                    # Clean up progress indicators
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    success_pages = len(text_content)
                    st.success(f"Successfully extracted {len(cleaned_text)} characters from {success_pages} pages")
                    
                    return cleaned_text
                    
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                
                if "PdfReadError" in str(type(e)) or "invalid" in str(e).lower():
                    st.error("PDF file appears to be corrupted, encrypted, or in an unsupported format.")
                else:
                    st.error(f"Error reading PDF: {str(e)}")
                return None
                
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {str(e)}")
            return None
    
    def get_paper_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get basic statistics about the paper text.
        """
        if not text or not text.strip():
            return {
                'character_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'avg_words_per_sentence': 0,
                'estimated_reading_time': 0
            }
        
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
        estimated_reading_time = len(words) / 200  # Assuming 200 words per minute
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_words_per_sentence': round(avg_words_per_sentence, 1),
            'estimated_reading_time': round(estimated_reading_time, 1)
        }
    
    def cleanup_temp_files(self):
        """Clean up temporary files and directory."""
        try:
            # Clean up individual tracked files
            cleaned_files = 0
            for file_path in self.temp_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        cleaned_files += 1
                except Exception as e:
                    st.warning(f"Could not remove {file_path}: {str(e)}")
            
            # Clean up temp directory
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                    st.info(f"Cleaned up temporary directory and {cleaned_files} files")
                except Exception as e:
                    st.warning(f"Could not remove temp directory: {str(e)}")
            
            self.temp_files.clear()
            
        except Exception as e:
            st.warning(f"Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.cleanup_temp_files()
        except:
            pass  # Ignore errors during destruction

def test_paper_processor():
    """Test the paper processor functionality."""
    st.title("Paper Processor Test")
    
    processor = PaperProcessor()
    
    # Test with a simple arXiv paper
    test_url = "https://arxiv.org/abs/1706.03762"
    
    if st.button("Test arXiv Download"):
        st.info(f"Testing with: {test_url}")
        
        metadata = processor.download_arxiv_paper(test_url)
        if metadata:
            st.success("Download successful!")
            st.json(metadata)
            
            # Test text extraction
            if metadata.get('local_path') and os.path.exists(metadata['local_path']):
                text = processor.extract_text_from_pdf(metadata['local_path'])
                if text:
                    st.success("Text extraction successful!")
                    stats = processor.get_paper_statistics(text)
                    st.json(stats)
                    st.text_area("Sample text:", text[:1000], height=200)
                else:
                    st.error("Text extraction failed")
            else:
                st.error("PDF file not accessible")
        else:
            st.error("Download failed")
    
    # Cleanup button
    if st.button("Cleanup Temp Files"):
        processor.cleanup_temp_files()

if __name__ == "__main__":
    test_paper_processor()