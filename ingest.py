"""
File ingestion and embeddings module.
Handles document parsing, chunking, and vector storage.
"""
import os
import json
import shutil
from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from utils.loaders import load_document
from utils.hashing import calculate_file_hash


class DocumentIngestor:
    """Handles document ingestion and vector storage."""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        Initialize the document ingestor.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            embedding_model: Ollama embedding model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings (OSS - Ollama)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Hash tracking file
        self.hash_file = os.path.join(persist_directory, "ingested_hashes.json")
        self.ingested_hashes = self._load_hashes()
        
        # Initialize or load vector store
        self.vectorstore = self._initialize_vectorstore()
    
    def _load_hashes(self) -> dict:
        """Load previously ingested file hashes."""
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_hashes(self):
        """Save ingested file hashes."""
        os.makedirs(os.path.dirname(self.hash_file), exist_ok=True)
        with open(self.hash_file, 'w') as f:
            json.dump(self.ingested_hashes, f, indent=2)
    
    def _initialize_vectorstore(self) -> Optional[Chroma]:
        """Initialize or load the vector store."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            # Load existing vector store
            try:
                return Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except Exception:
                # If loading fails, create new one
                return None
        return None
    
    def is_file_ingested(self, file_path: str) -> bool:
        """
        Check if a file has already been ingested.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file has been ingested, False otherwise
        """
        file_hash = calculate_file_hash(file_path)
        return file_hash in self.ingested_hashes
    
    def ingest_file(self, file_path: str, filename: str) -> dict:
        """
        Ingest a single file into the vector store.
        
        Args:
            file_path: Path to the file
            filename: Original filename for metadata
            
        Returns:
            Dictionary with ingestion results
        """
        # Check if already ingested
        file_hash = calculate_file_hash(file_path)
        if file_hash in self.ingested_hashes:
            return {
                "status": "skipped",
                "filename": filename,
                "message": "File already ingested (duplicate)"
            }
        
        try:
            # Load document
            documents = load_document(file_path)
            
            # Add filename to metadata
            for doc in documents:
                doc.metadata['source_file'] = filename
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                return {
                    "status": "error",
                    "filename": filename,
                    "message": "No content extracted from file"
                }
            
            # Create or update vector store
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                self.vectorstore.add_documents(chunks)
            
            # Save hash
            self.ingested_hashes[file_hash] = filename
            self._save_hashes()
            
            return {
                "status": "success",
                "filename": filename,
                "chunks": len(chunks),
                "message": f"Successfully ingested {len(chunks)} chunks"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "filename": filename,
                "message": f"Error: {str(e)}"
            }
    
    def clear_database(self):
        """Clear the entire vector database and hash tracking."""
        # Clear vector store
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except Exception:
                pass
            self.vectorstore = None
        
        # Clear persisted files
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
            except Exception:
                pass
        
        # Clear hashes
        self.ingested_hashes = {}
        self._save_hashes()
    
    def get_vectorstore(self) -> Optional[Chroma]:
        """Get the vector store instance."""
        return self.vectorstore
    
    def has_documents(self) -> bool:
        """Check if the vector store has any documents."""
        return self.vectorstore is not None and len(self.ingested_hashes) > 0
