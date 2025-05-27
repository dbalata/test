"""Tests for the actual exceptions module with proper mocking."""
import sys
import types
from unittest.mock import patch, MagicMock
import pytest

# Create a mock logger module
mock_logger_module = types.ModuleType('src.utils.logger')
mock_logger = MagicMock()
mock_logger_module.logger = mock_logger
mock_logger_module.get_logger = lambda name: mock_logger
mock_logger_module.setup_logger = lambda name, log_level=None, log_file=None: mock_logger

# Create a mock settings module
mock_settings_module = types.ModuleType('src.config.settings')
mock_settings = MagicMock()
mock_settings.app = MagicMock()
mock_settings.app.log_level = "INFO"
mock_settings.openrouter_api_key = None
mock_settings.openrouter_base_url = "http://test-api"
mock_settings_module.settings = mock_settings
mock_settings_module.Settings = MagicMock(return_value=mock_settings)

# Patch the modules
sys.modules['src.utils.logger'] = mock_logger_module
sys.modules['src.config.settings'] = mock_settings_module

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

def test_actual_langchain_qa_error_initialization():
    """Test basic initialization of the actual LangChainQAError."""
    error = LangChainQAError("Test error")
    assert str(error) == "Test error"
    assert error.details == {}
    assert error.cause is None

def test_actual_langchain_qa_error_with_details():
    """Test actual LangChainQAError with additional details."""
    details = {"key": "value", "count": 42}
    error = LangChainQAError("Test error", details=details)
    assert error.details == details

def test_actual_langchain_qa_error_with_cause():
    """Test actual LangChainQAError with a cause."""
    cause = ValueError("Original error")
    error = LangChainQAError("Test error", cause=cause)
    assert error.cause is cause

@patch('src.utils.exceptions.logger')
def test_actual_langchain_qa_error_logging(mock_logger):
    """Test that actual LangChainQAError logs errors correctly."""
    details = {"key": "value"}
    cause = ValueError("Original error")
    
    with pytest.raises(LangChainQAError) as exc_info:
        raise LangChainQAError("Test error", details=details, cause=cause)
    
    # Check the error was logged
    mock_logger.error.assert_called_once()
    log_message = mock_logger.error.call_args[0][0]
    assert "Test error" in log_message
    assert "Details: {'key': 'value'}" in log_message
    assert "Caused by: Original error" in log_message
    
    # Check the exception was chained correctly
    assert exc_info.value.__cause__ is cause

def test_actual_exception_hierarchy():
    """Test that all actual custom exceptions inherit from LangChainQAError."""
    from src.utils.exceptions import (
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
