"""
Utility functions and helpers for the LangChain Q&A System.

This package contains various utility modules that provide common functionality
used throughout the application.
"""
from .logger import logger, get_logger, setup_logger
from .exceptions import (
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

__all__ = [
    # Logger
    "logger",
    "get_logger",
    "setup_logger",
    
    # Exceptions
    "LangChainQAError",
    "ConfigurationError",
    "DocumentProcessingError",
    "VectorStoreError",
    "LLMError",
    "RetrievalError",
    "GenerationError",
    "ValidationError",
    "AuthenticationError",
    "RateLimitError",
    "ResourceNotFoundError",
    "UnsupportedOperationError",
]
