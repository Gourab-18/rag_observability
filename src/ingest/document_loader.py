# helps load documents from various file formats

from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    
    @staticmethod
    def load_pdf(file_path: Path) -> str:

# loads from pdf and converts to text
# tries to use multiple libraries if one fails then used another
        try:
            import pdfplumber
        except ImportError:
            try:
                import PyPDF2
            except ImportError:
                raise ImportError(
                    "PDF library required. Install with: pip install pdfplumber or pip install PyPDF2"
                )
            else:
                # Use PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text.strip()
        else:
            # Use pdfplumber (preferred)
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
    
    @staticmethod
    def load_text(file_path: Path) -> str:
# from saved file loads text
        # Try UTF-8 first, fallback to latin-1 if needed
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, read as binary and decode with errors='replace'
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')
    
    @staticmethod
    def load_document(file_path: Path) -> Dict[str, Any]:

# decides which type of file it is and calls above 2 functions
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            text = DocumentLoader.load_pdf(file_path)
        elif suffix in ['.txt', '.md', '.markdown']:
            text = DocumentLoader.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: .pdf, .txt, .md")
        
        # Get file metadata
        file_stat = file_path.stat()
        
        metadata = {
            'file_path': str(file_path.absolute()),
            'file_name': file_path.name,
            'file_type': suffix,
            'file_size': file_stat.st_size,
            'file_extension': suffix
        }
        
        # returns entire text and metadata object
        return {
            'text': text,
            'metadata': metadata
        }

