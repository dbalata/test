"""
Document processing module for handling various document types and creating vector stores.
"""

import os
import tempfile
from typing import List, Any
import streamlit as st
from io import BytesIO

from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

import requests
from bs4 import BeautifulSoup


class DocumentProcessor:
    """Handles document ingestion, processing, and vector store creation."""
    
    def __init__(self):
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = OpenAIEmbeddings()
    
    def add_file(self, uploaded_file) -> None:
        """Add an uploaded file to the document collection."""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load document based on file type
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.endswith(('.txt', '.md')):
                loader = TextLoader(tmp_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.name}")
            
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    'source': uploaded_file.name,
                    'type': 'uploaded_file'
                })
            
            self.documents.extend(docs)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            raise Exception(f"Error processing file {uploaded_file.name}: {str(e)}")
    
    def add_url(self, url: str) -> None:
        """Add content from a URL to the document collection."""
        try:
            # Use WebBaseLoader for basic web scraping
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    'source': url,
                    'type': 'web_content'
                })
            
            self.documents.extend(docs)
            
        except Exception as e:
            # Fallback to manual scraping
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Create document
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': url,
                        'type': 'web_content',
                        'title': soup.title.string if soup.title else 'Unknown'
                    }
                )
                
                self.documents.append(doc)
                
            except Exception as e2:
                raise Exception(f"Error scraping URL {url}: {str(e2)}")
    
    def add_text(self, text: str, source: str = "manual_input") -> None:
        """Add raw text to the document collection."""
        doc = Document(
            page_content=text,
            metadata={
                'source': source,
                'type': 'manual_text'
            }
        )
        self.documents.append(doc)
    
    def create_vector_store(self) -> Chroma:
        """Create a vector store from the collected documents."""
        if not self.documents:
            raise ValueError("No documents to process. Please add documents first.")
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(self.documents)
        
        if not texts:
            raise ValueError("No text chunks created from documents.")
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return vector_store
    
    def get_document_info(self) -> dict:
        """Get information about the loaded documents."""
        info = {
            'total_documents': len(self.documents),
            'sources': [],
            'types': {}
        }
        
        for doc in self.documents:
            source = doc.metadata.get('source', 'Unknown')
            doc_type = doc.metadata.get('type', 'Unknown')
            
            if source not in info['sources']:
                info['sources'].append(source)
            
            if doc_type in info['types']:
                info['types'][doc_type] += 1
            else:
                info['types'][doc_type] = 1
        
        return info
