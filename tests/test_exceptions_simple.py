"""Simple tests for the exceptions module."""
import pytest

# Mock the logger to avoid initialization issues
class MockLogger:
    def error(self, *args, **kwargs):
        pass

class MockSettings:
    def __init__(self):
        self.openrouter_api_key = None
        self.openrouter_base_url = "http://test-api"

# Mock the logger module
import sys
import types
logger_module = types.ModuleType('src.utils.logger')
logger_module.get_logger = lambda name: MockLogger()
sys.modules['src.utils.logger'] = logger_module

# Mock the settings module
settings_module = types.ModuleType('src.config.settings')
settings_module.Settings = MockSettings
settings_module.settings = MockSettings()
sys.modules['src.config.settings'] = settings_module

# Now import the exceptions
from src.utils.exceptions import (
    LangChainQAError,
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
)

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
