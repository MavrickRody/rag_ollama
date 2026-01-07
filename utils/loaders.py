"""
Open-source document loaders.
All loaders use OSS-compliant libraries only.
"""
from typing import List
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document


def load_pdf(file_path: str) -> List[Document]:
    """
    Load PDF file using PyPDFLoader (OSS).
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        List of Document objects
    """
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_docx(file_path: str) -> List[Document]:
    """
    Load DOCX file using Docx2txtLoader (OSS).
    
    Args:
        file_path: Path to DOCX file
        
    Returns:
        List of Document objects
    """
    loader = Docx2txtLoader(file_path)
    return loader.load()


def load_csv(file_path: str) -> List[Document]:
    """
    Load CSV file using CSVLoader (OSS).
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        List of Document objects
    """
    loader = CSVLoader(file_path)
    return loader.load()


def load_xlsx(file_path: str) -> List[Document]:
    """
    Load XLSX file using UnstructuredExcelLoader (OSS).
    
    Args:
        file_path: Path to XLSX file
        
    Returns:
        List of Document objects
    """
    loader = UnstructuredExcelLoader(file_path)
    return loader.load()


def load_document(file_path: str) -> List[Document]:
    """
    Load document based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of Document objects
        
    Raises:
        ValueError: If file type is not supported
    """
    extension = Path(file_path).suffix.lower().lstrip('.')
    
    loaders = {
        'pdf': load_pdf,
        'docx': load_docx,
        'csv': load_csv,
        'xlsx': load_xlsx,
        'xls': load_xlsx,
    }
    
    if extension not in loaders:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(loaders.keys())}"
        )
    
    return loaders[extension](file_path)
