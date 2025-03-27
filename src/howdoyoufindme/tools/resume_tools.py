from langchain.tools import BaseTool
from langchain.tools.base import Tool
from langchain.utilities import BingSearchAPIWrapper
from typing import Optional, Dict, Any, List, Callable, Type, ClassVar
import requests
from bs4 import BeautifulSoup
import os
import json
import tempfile
from pydantic import BaseModel, Field
import logging
import mimetypes

# Add logger instantiation here
logger = logging.getLogger(__name__)

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyPDF2 not installed. PDF support will be limited.")

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    logging.warning("python-docx not installed. DOCX support will be limited.")

class ResumeDocumentInput(BaseModel):
    file_path: str = Field(..., description="Path to the resume file to analyze")

class JobSearchInput(BaseModel):
    query: str = Field(..., description="Search query for job information")

class CacheInput(BaseModel):
    action: str = Field(..., description="Action to perform: 'store' or 'retrieve'")
    key: Optional[str] = Field(None, description="Key for identifying stored data")
    data: Optional[Dict[str, Any]] = Field(None, description="Data to store (only for 'store' action)")

def get_file_content(file_path: str) -> str:
    """Extract text from various file formats"""
    logger.info(f"[get_file_content] Processing file: {file_path}")
    if not os.path.exists(file_path):
        error_msg = f"Error: File {file_path} not found"
        logger.error(f"[get_file_content] {error_msg}")
        return error_msg
    
    # Determine file type
    file_type = mimetypes.guess_type(file_path)[0]
    
    # Handle different file types
    if file_type == 'application/pdf' and PDF_SUPPORT:
        logger.info("[get_file_content] Attempting PDF extraction.")
        result = extract_pdf_text(file_path)
        if result.startswith("Error:"):
             logger.warning(f"[get_file_content] PDF extraction failed: {result}")
        return result
    elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'] and DOCX_SUPPORT:
        return extract_docx_text(file_path)
    else:
        # Fallback: try to read as text
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # If not text, just return info about the file
            return f"File {os.path.basename(file_path)} exists but couldn't be read as text. File type: {file_type or 'unknown'}"

def extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting PDF text: {str(e)}")
        return f"Error extracting text from PDF: {str(e)}"

def extract_docx_text(file_path: str) -> str:
    """Extract text from a DOCX file"""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logging.error(f"Error extracting DOCX text: {str(e)}")
        return f"Error extracting text from DOCX: {str(e)}"

def resume_document_tool(file_path: str) -> str:
    """Extract text and structure from resume files in PDF or Word format"""
    logger.info(f"[resume_document_tool] Attempting to read file: {file_path}")

    if not os.path.exists(file_path):
        error_msg = f"Error: File not found at path: {file_path}"
        logger.error(f"[resume_document_tool] {error_msg}")
        return error_msg

    try:
        # Get file info
        file_size = os.path.getsize(file_path)
        file_type = mimetypes.guess_type(file_path)[0] or "unknown"
        logger.info(f"[resume_document_tool] File exists: {os.path.basename(file_path)}, Size: {file_size} bytes, Type: {file_type}")

        # Extract content based on file type
        content = get_file_content(file_path)

        # Check if content extraction failed (returned an error message)
        if content.startswith("Error:"):
             logger.error(f"[resume_document_tool] Content extraction failed: {content}")
             return content

        # Log success and truncate if necessary
        logger.info(f"[resume_document_tool] Successfully extracted content (length: {len(content)}).")
        if len(content) > 10000:
            content = content[:10000] + "... [content truncated]"
            logger.info("[resume_document_tool] Content truncated.")

        # Return the actual content if successful
        return content

    except Exception as e:
        error_msg = f"Error: Unexpected exception in resume_document_tool for {file_path}: {str(e)}"
        logger.error(f"[resume_document_tool] {error_msg}", exc_info=True)
        return error_msg

class JobSearchTool:
    """Tool for searching job information online"""
    bing_search = None
    
    def __init__(self):
        self.bing_search = BingSearchAPIWrapper(
            bing_subscription_key=os.getenv('BING_SUBSCRIPTION_KEY'),
            bing_search_url="https://api.bing.microsoft.com/v7.0/search"
        )
    
    def run(self, query: str) -> str:
        """Search for job-related information"""
        try:
            results = self.bing_search.run(query)
            return results
        except Exception as e:
            logging.error(f"Error performing search: {str(e)}", exc_info=True)
            return f"Error performing search: {str(e)}"

def job_search_tool(query: str) -> str:
    """Search for company and job information online"""
    search_tool = JobSearchTool()
    return search_tool.run(query)

class CacheStorageTool:
    """Tool for storing and retrieving cached analysis results"""
    
    def __init__(self):
        self.cache_dir = os.path.join(tempfile.gettempdir(), "resume_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def run(self, action: str, key: str = None, data: Dict[str, Any] = None) -> str:
        """Store or retrieve cached data"""
        if action == "store" and key and data:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            return f"Data stored in cache with key: {key}"
            
        elif action == "retrieve" and key:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.dumps(json.load(f), indent=2)
            return f"No cache found for key: {key}"
            
        return "Invalid action. Use 'store' or 'retrieve' with appropriate parameters."

def cache_storage_tool(action: str, key: str = None, data: Dict[str, Any] = None) -> str:
    """Store and retrieve cached analysis results"""
    cache_tool = CacheStorageTool()
    return cache_tool.run(action, key, data)

# Create LangChain Tool instances
resume_tool = Tool(
    name="resume_document_tool",
    func=resume_document_tool,
    description="Extract text and structure from resume files in PDF or Word format",
    args_schema=ResumeDocumentInput
)

job_search_tool = Tool(
    name="job_search_tool",
    func=job_search_tool,
    description="Search for company and job information online",
    args_schema=JobSearchInput
)

cache_tool = Tool(
    name="cache_storage_tool",
    func=cache_storage_tool,
    description="Store and retrieve cached analysis results",
    args_schema=CacheInput
) 