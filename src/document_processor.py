"""
Document processing module for handling various document types and creating vector stores.

This module provides the DocumentProcessor class which handles loading, processing,
and storing documents in a vector database for efficient similarity search.
"""
from __future__ import annotations

import os
import tempfile
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Literal, overload
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import settings
from .utils import (
    logger,
    DocumentProcessingError,
    ValidationError,
    ResourceNotFoundError,
    VectorStoreError,
)

# Configure mimetypes
mimetypes.add_type('text/markdown', '.md')


class DocumentProcessor:
    """
    Handles document ingestion, processing, and vector store creation.
    
    This class provides methods to load documents from various sources (files, URLs, raw text),
    process them into chunks, and store them in a vector database for efficient similarity search.
    """
    
    def __init__(
        self,
        embedding_model: Optional[Embeddings] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        persist_directory: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the DocumentProcessor.
        
        Args:
            embedding_model: The embedding model to use. If None, uses default from settings.
            chunk_size: Size of text chunks. If None, uses default from settings.
            chunk_overlap: Overlap between chunks. If None, uses default from settings.
            persist_directory: Directory to persist the vector store. If None, uses default from settings.
        """
        self.documents: List[Document] = []
        
        # Initialize text splitter
        self.chunk_size = chunk_size or settings.vector_store.chunk_size
        self.chunk_overlap = chunk_overlap or settings.vector_store.chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        # Initialize embeddings
        self.embedding_model = embedding_model or self._get_default_embeddings()
        
        # Set up persistence
        self.persist_directory = Path(
            persist_directory or settings.vector_store.persist_directory
        )
    
    def _get_default_embeddings(self) -> Embeddings:
        """Get the default embedding model based on configuration."""
        model_name = settings.vector_store.embedding_model
        
        if "text-embedding" in model_name:
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=settings.llm.api_key,
                openai_api_base=settings.llm.base_url,
            )
        else:
            return HuggingFaceEmbeddings(model_name=model_name)
    
    def add_file(self, file_path: Union[str, Path], **metadata: Any) -> None:
        """Add a file to the document collection.
        
        Args:
            file_path: Path to the file to add.
            **metadata: Additional metadata to include with the document.
            
        Raises:
            ResourceNotFoundError: If the file does not exist.
            DocumentProcessingError: If there is an error processing the file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ResourceNotFoundError(f"File not found: {file_path}")
        
        try:
            # Determine loader based on file extension
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in ('.txt', '.md'):
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:
                # Try using UnstructuredFileLoader for other file types
                loader = UnstructuredFileLoader(str(file_path))
            
            docs = loader.load()
            
            # Add metadata to each document
            for doc in docs:
                doc.metadata.update({
                    'source': str(file_path),
                    'type': 'file',
                    'file_name': file_path.name,
                    'file_type': file_path.suffix.lower().lstrip('.'),
                    **metadata,
                })
            
            self.documents.extend(docs)
            logger.info(f"Successfully loaded {len(docs)} documents from {file_path}")
            
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
    
    def add_uploaded_file(self, uploaded_file: Any, **metadata: Any) -> None:
        """Add an uploaded file (e.g., from Streamlit file_uploader) to the document collection.
        
        Args:
            uploaded_file: The uploaded file object (e.g., from Streamlit).
            **metadata: Additional metadata to include with the document.
            
        Raises:
            DocumentProcessingError: If there is an error processing the file.
        """
        try:
            # Save uploaded file to temporary location
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Process the temporary file
                self.add_file(tmp_path, **metadata)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")
                    
        except Exception as e:
            error_msg = f"Error processing uploaded file {uploaded_file.name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
    
    def add_url(self, url: str, **metadata: Any) -> None:
        """Add content from a URL to the document collection.
        
        Args:
            url: The URL to fetch content from.
            **metadata: Additional metadata to include with the document.
            
        Raises:
            ValidationError: If the URL is invalid.
            DocumentProcessingError: If there is an error fetching or processing the URL.
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValidationError(f"Invalid URL: {url}")
            
            logger.info(f"Fetching content from URL: {url}")
            
            # First try using WebBaseLoader
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
            except Exception as e:
                logger.warning(
                    f"WebBaseLoader failed for {url}, falling back to manual scraping: {e}"
                )
                docs = self._scrape_url_manually(url)
            
            # Add metadata to each document
            for doc in docs:
                doc.metadata.update({
                    'source': url,
                    'type': 'web_content',
                    'domain': parsed_url.netloc,
                    **metadata,
                })
            
            self.documents.extend(docs)
            logger.info(f"Successfully loaded {len(docs)} documents from {url}")
            
        except Exception as e:
            error_msg = f"Error processing URL {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
    
    def _scrape_url_manually(self, url: str) -> List[Document]:
        """Manually scrape content from a URL.
        
        This is a fallback method when WebBaseLoader fails.
        """
        try:
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Get title
            title = soup.title.string if soup.title else "Untitled Document"
            
            # Get main content (try to find the main content area)
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Create document
            doc = Document(
                page_content=clean_text,
                metadata={
                    'title': title,
                }
            )
            
            return [doc]
            
        except Exception as e:
            logger.error(f"Manual scraping failed for {url}: {str(e)}")
            raise DocumentProcessingError(f"Failed to scrape URL {url}: {str(e)}") from e
    
    def add_text(self, text: str, source: str = "manual_input", **metadata: Any) -> None:
        """Add raw text to the document collection.
        
        Args:
            text: The text content to add.
            source: Source identifier for the text.
            **metadata: Additional metadata to include with the document.
        """
        if not text or not isinstance(text, str):
            raise ValidationError("Text must be a non-empty string")
        
        doc = Document(
            page_content=text,
            metadata={
                'source': source,
                'type': 'text',
                **metadata,
            }
        )
        self.documents.append(doc)
        logger.info(f"Added text document from source: {source}")
    
    def create_vector_store(self) -> Union[Chroma, FAISS]:
        """Create a vector store from the collected documents.
        
        Returns:
            A vector store containing the document embeddings.
            
        Raises:
            ValueError: If no documents have been added.
            VectorStoreError: If there is an error creating the vector store.
        """
        if not self.documents:
            raise ValueError("No documents to process. Please add documents first.")
        
        try:
            # Split documents into chunks
            logger.info(f"Splitting {len(self.documents)} documents into chunks...")
            texts = self.text_splitter.split_documents(self.documents)
            
            if not texts:
                raise ValueError("No text chunks created from documents. The documents may be empty or too short.")
            
            logger.info(f"Creating vector store with {len(texts)} chunks...")
            
            # Ensure persist directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Create vector store based on configured type
            if settings.vector_store.type.lower() == 'faiss':
                from langchain_community.vectorstores import FAISS
                vector_store = FAISS.from_documents(
                    documents=texts,
                    embedding=self.embedding_model,
                )
                # FAISS requires explicit save
                faiss_path = self.persist_directory / "faiss_index"
                vector_store.save_local(str(faiss_path))
            else:  # Default to Chroma
                vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=self.embedding_model,
                    persist_directory=str(self.persist_directory),
                    collection_name=settings.vector_store.collection_name,
                )
            
            logger.info(f"Vector store created successfully at {self.persist_directory}")
            return vector_store
            
        except Exception as e:
            error_msg = f"Error creating vector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg) from e
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the loaded documents.
        
        Returns:
            A dictionary containing document statistics and metadata.
        """
        info: Dict[str, Any] = {
            'total_documents': len(self.documents),
            'total_chunks': 0,
            'sources': [],
            'types': {},
            'size_bytes': 0,
        }
        
        source_map: Dict[str, int] = {}
        type_map: Dict[str, int] = {}
        
        for doc in self.documents:
            # Track sources
            source = doc.metadata.get('source', 'unknown')
            source_map[source] = source_map.get(source, 0) + 1
            
            # Track document types
            doc_type = doc.metadata.get('type', 'unknown')
            type_map[doc_type] = type_map.get(doc_type, 0) + 1
            
            # Calculate total size
            info['size_bytes'] += len(doc.page_content.encode('utf-8'))
        
        # Calculate chunks if text splitter is available
        if self.documents:
            try:
                chunks = self.text_splitter.split_documents(self.documents)
                info['total_chunks'] = len(chunks)
                info['avg_chunk_size'] = sum(len(c.page_content) for c in chunks) / max(1, len(chunks))
            except Exception as e:
                logger.warning(f"Error calculating chunk stats: {e}")
        
        info['sources'] = [
            {'source': src, 'count': count}
            for src, count in sorted(source_map.items(), key=lambda x: x[1], reverse=True)
        ]
        
        info['types'] = type_map
        info['size_mb'] = info['size_bytes'] / (1024 * 1024)
        
        return info
    
    def clear_documents(self) -> None:
        """Clear all loaded documents from the processor."""
        self.documents.clear()
        logger.info("Cleared all documents from the processor")
