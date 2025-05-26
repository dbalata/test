"""
Custom exceptions for the LangChain Q&A System.

This module defines a hierarchy of custom exceptions that can be raised throughout
the application. Using specific exception types makes error handling more precise
and provides better error messages to users.
"""
from typing import Any, Dict, Optional

from .logger import get_logger

logger = get_logger(__name__)

class LangChainQAError(Exception):
    """Base exception for all LangChain Q&A application errors."""
    
    def __init__(
        self,
        message: str = "An error occurred in the LangChain Q&A System",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Additional error details for debugging
            cause: The underlying exception that caused this error, if any
        """
        self.message = message
        self.details = details or {}
        self.cause = cause
        
        # Log the error
        log_message = message
        if details:
            log_message += f"\nDetails: {details}"
        if cause:
            log_message += f"\nCaused by: {str(cause)}"
        
        logger.error(log_message, exc_info=True)
        
        super().__init__(message)


class ConfigurationError(LangChainQAError):
    """Raised when there is a configuration error."""
    pass


class DocumentProcessingError(LangChainQAError):
    """Raised when there is an error processing documents."""
    pass


class VectorStoreError(LangChainQAError):
    """Raised when there is an error with the vector store."""
    pass


class LLMError(LangChainQAError):
    """Raised when there is an error with the LLM."""
    pass


class RetrievalError(LangChainQAError):
    """Raised when there is an error during document retrieval."""
    pass


class GenerationError(LangChainQAError):
    """Raised when there is an error during text generation."""
    pass


class ValidationError(LangChainQAError, ValueError):
    """Raised when input validation fails."""
    pass


class AuthenticationError(LangChainQAError, PermissionError):
    """Raised when authentication or authorization fails."""
    pass


class RateLimitError(LangChainQAError):
    """Raised when API rate limits are exceeded."""
    pass


class ResourceNotFoundError(LangChainQAError, FileNotFoundError):
    """Raised when a requested resource is not found."""
    pass


class UnsupportedOperationError(LangChainQAError, NotImplementedError):
    """Raised when an operation is not supported."""
    pass


__all__ = [
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
