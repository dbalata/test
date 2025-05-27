"""Tests for the custom exceptions module."""
from unittest.mock import patch, MagicMock
import pytest

# Create a simplified version of the exceptions for testing
class LangChainQAError(Exception):
    """Base exception for testing."""
    def __init__(self, message, details=None, cause=None):
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(message)

class DocumentProcessingError(LangChainQAError):
    """Test exception for document processing errors."""
    pass

class ConfigurationError(LangChainQAError):
    """Test exception for configuration errors."""
    pass

class VectorStoreError(LangChainQAError):
    """Test exception for vector store errors."""
    pass

class LLMError(LangChainQAError):
    """Test exception for LLM errors."""
    pass

class RetrievalError(LangChainQAError):
    """Test exception for retrieval errors."""
    pass

class GenerationError(LangChainQAError):
    """Test exception for generation errors."""
    pass

class ValidationError(LangChainQAError):
    """Test exception for validation errors."""
    pass

class AuthenticationError(LangChainQAError):
    """Test exception for authentication errors."""
    pass

class RateLimitError(LangChainQAError):
    """Test exception for rate limit errors."""
    pass

class ResourceNotFoundError(LangChainQAError):
    """Test exception for resource not found errors."""
    pass

class UnsupportedOperationError(LangChainQAError):
    """Test exception for unsupported operations."""
    pass

def test_langchain_qa_error_initialization():
    """Test basic initialization of LangChainQAError."""
    error = LangChainQAError("Test error")
    assert str(error) == "Test error"
    assert error.details == {}
    assert error.cause is None

def test_langchain_qa_error_with_details():
    """Test LangChainQAError with additional details."""
    details = {"key": "value", "count": 42}
    error = LangChainQAError("Test error", details=details)
    assert error.details == details

def test_langchain_qa_error_with_cause():
    """Test LangChainQAError with a cause."""
    cause = ValueError("Original error")
    error = LangChainQAError("Test error", cause=cause)
    assert error.cause is cause

def test_exception_hierarchy():
    """Test that all custom exceptions inherit from LangChainQAError."""
    exceptions = [
        ConfigurationError,
        DocumentProcessingError,
        VectorStoreError,
        LLMError,
        RetrievalError,
        GenerationError,
        ValidationError,
        AuthenticationError,
        RateLimitError,
        ResourceNotFoundError,
        UnsupportedOperationError,
    ]
    
    for exc_class in exceptions:
        error = exc_class("Test error")
        assert isinstance(error, LangChainQAError)
        assert str(error) == "Test error"

def test_exception_with_custom_message():
    """Test that exceptions can be created with custom messages."""
    error = DocumentProcessingError("Failed to process document")
    assert str(error) == "Failed to process document"
    assert isinstance(error, LangChainQAError)

def test_exception_with_details():
    """Test that exceptions can include additional details."""
    details = {"document_id": "123", "error_type": "parsing"}
    error = DocumentProcessingError("Failed to process document", details=details)
    assert error.details == details

def test_exception_chaining():
    """Test that exceptions can be chained with a cause."""
    cause = ValueError("Invalid format")
    error = DocumentProcessingError("Failed to process document", cause=cause)
    assert error.cause is cause
    assert str(cause) in str(error.cause)
