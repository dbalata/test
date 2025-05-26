"""
Tests for the DocumentProcessor class.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pytest
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.document_processor import DocumentProcessor
from src.utils import (
    DocumentProcessingError,
    ValidationError,
    ResourceNotFoundError,
    VectorStoreError,
)

# Mock response for OpenAI embeddings
MOCK_EMBEDDINGS = [[0.1] * 1536]  # Mock embedding vector for testing


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield DocumentProcessor(persist_directory=temp_dir)

    @pytest.fixture
    def sample_text_file(self):
        """Create a sample text file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.")
            f.flush()
            yield f.name
        if os.path.exists(f.name):
            os.unlink(f.name)

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings for testing."""
        with patch('langchain_community.embeddings.OpenAIEmbeddings') as mock_embeddings:
            mock_instance = MagicMock()
            mock_instance.embed_documents.return_value = MOCK_EMBEDDINGS
            mock_instance.embed_query.return_value = MOCK_EMBEDDINGS[0]
            mock_embeddings.return_value = mock_instance
            yield mock_instance

    def test_add_text(self, processor):
        """Test adding text to the processor."""
        processor.add_text("Test content", source="test")
        assert len(processor.documents) == 1
        assert processor.documents[0].page_content == "Test content"
        assert processor.documents[0].metadata["source"] == "test"

    def test_add_text_invalid(self, processor):
        """Test adding invalid text raises ValidationError."""
        with pytest.raises(ValidationError):
            processor.add_text("")
        
        with pytest.raises(ValidationError):
            processor.add_text(123)  # type: ignore

    def test_add_file(self, processor, sample_text_file):
        """Test adding a file to the processor."""
        processor.add_file(sample_text_file)
        assert len(processor.documents) > 0
        assert "test" in processor.documents[0].page_content.lower()

    def test_add_file_not_found(self, processor):
        """Test adding a non-existent file raises ResourceNotFoundError."""
        with pytest.raises(ResourceNotFoundError):
            processor.add_file("/nonexistent/file.txt")

    @patch('requests.get')
    def test_add_url_success(self, mock_get, processor):
        """Test adding a URL successfully."""
        # Mock the response for the actual example.com content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <!doctype html>
        <html>
        <head>
            <title>Example Domain</title>
            <meta charset="utf-8" />
            <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
        </head>
        <body>
            <h1>Example Domain</h1>
            <p>This domain is for use in illustrative examples in documents.</p>
        </body>
        </html>
        """
        mock_get.return_value = mock_response

        # Mock the WebBaseLoader to return our test content
        with patch('langchain_community.document_loaders.web_base.WebBaseLoader') as mock_loader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [
                Document(
                    page_content="Example Domain\nThis domain is for use in illustrative examples in documents.",
                    metadata={"source": "http://example.com"}
                )
            ]
            mock_loader.return_value = mock_loader_instance

            # Test the URL
            processor.add_url("http://example.com")
            assert len(processor.documents) > 0
            assert "example domain" in processor.documents[0].page_content.lower()

    @patch('requests.get')
    def test_add_url_fallback_scraping(self, mock_get, processor):
        """Test fallback to manual scraping when WebBaseLoader fails."""
        # Mock the response for manual scraping
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = """
        <html>
            <head><title>Test Page</title></head>
            <body><p>Test content</p></body>
        </html>
        """
        mock_get.return_value = mock_response
        
        # Make WebBaseLoader raise an exception
        with patch('langchain_community.document_loaders.web_base.WebBaseLoader.load') as mock_loader:
            mock_loader.side_effect = Exception("Loader failed")
            processor.add_url("http://example.com")
            
        assert len(processor.documents) > 0
        assert "test content" in processor.documents[0].page_content.lower()

    def test_add_url_invalid(self, processor):
        """Test adding an invalid URL raises a validation error."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            processor.add_url("not-a-url")
        assert "Invalid URL" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValidationError)

    @patch('langchain_community.vectorstores.chroma.Chroma.from_documents')
    def test_create_vector_store_success(self, mock_chroma, processor, mock_embeddings):
        """Test creating a vector store successfully."""
        # Mock the Chroma from_documents method
        mock_chroma.return_value = MagicMock(spec=Chroma)
        
        # Add some test documents
        processor.add_text("Test document 1", source="test1")
        processor.add_text("Test document 2", source="test2")
        
        # Test Chroma backend
        vector_store = processor.create_vector_store()
        
        # Verify the vector store was created
        assert vector_store is not None
        mock_chroma.assert_called_once()
        
        # Verify the persist directory was created
        assert processor.persist_directory.exists()

    @patch('langchain_community.vectorstores.chroma.Chroma.from_documents')
    def test_create_vector_store_error(self, mock_chroma, processor, mock_embeddings):
        """Test error handling when creating a vector store fails."""
        # Configure the mock to raise an exception
        mock_chroma.side_effect = Exception("Test error")
        
        # Add a test document
        processor.add_text("Test document", source="test")
        
        # Should raise VectorStoreError
        with pytest.raises(VectorStoreError, match="Error creating vector store"):
            processor.create_vector_store()

    def test_get_document_info(self, processor):
        """Test getting document information."""
        # Add test documents
        processor.add_text("Doc 1", source="test1", type="test")
        processor.add_text("Doc 2", source="test2", type="test")
        processor.add_text("Doc 3", source="test1", type="test")
        
        info = processor.get_document_info()
        
        assert info["total_documents"] == 3
        assert info["types"]["test"] == 3
        assert len(info["sources"]) == 2  # test1 and test2
        assert info["size_bytes"] > 0

    def test_clear_documents(self, processor):
        """Test clearing all documents from the processor."""
        # Add a document
        processor.add_text("Test document", source="test")
        assert len(processor.documents) == 1
        
        # Clear documents
        processor.clear_documents()
        assert len(processor.documents) == 0
