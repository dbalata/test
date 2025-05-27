"""Custom exceptions for the LangChain Q&A System.

This module defines a hierarchy of custom exceptions that can be raised throughout
the application. Using specific exception types makes error handling more precise
and provides better error messages to users.
"""
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar, Union

from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T', bound='ErrorCode')

class ErrorCode(Enum):
    """Error codes for different types of exceptions."""
    UNKNOWN_ERROR = ("0000", "An unknown error occurred")
    CONFIGURATION_ERROR = ("1000", "Configuration error")
    DOCUMENT_PROCESSING_ERROR = ("2000", "Document processing error")
    VECTOR_STORE_ERROR = ("3000", "Vector store error")
    LLM_ERROR = ("4000", "LLM error")
    RETRIEVAL_ERROR = ("5000", "Retrieval error")
    GENERATION_ERROR = ("6000", "Generation error")
    VALIDATION_ERROR = ("7000", "Validation error")
    AUTHENTICATION_ERROR = ("8000", "Authentication error")
    RATE_LIMIT_ERROR = ("8001", "Rate limit exceeded")
    RESOURCE_NOT_FOUND = ("9000", "Resource not found")
    UNSUPPORTED_OPERATION = ("9001", "Unsupported operation")

    def __init__(self, code: str, message: str) -> None:
        self._code = code
        self._message = message

    @property
    def code(self) -> str:
        return self._code

    @property
    def message(self) -> str:
        return self._message

    @classmethod
    def from_code(cls: Type[T], code: Union[str, int]) -> 'ErrorCode':
        """Get error code enum from code string or integer."""
        code_str = str(code).zfill(4)
        for error in cls:
            if error.code == code_str:
                return error
        return cls.UNKNOWN_ERROR

class LangChainQAError(Exception):
    """Base exception for all LangChain Q&A application errors."""
    
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR
    
    def __init__(
        self,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None,
    ) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error message. If None, uses the default message
                   from the error code.
            details: Additional error details for debugging
            cause: The underlying exception that caused this error, if any
            error_code: Specific error code for this exception
        """
        self.error_code = error_code or self.error_code
        self.message = message or self.error_code.message
        self.details = details or {}
        self.cause = cause
        
        # Format the full message with error code
        full_message = f"[{self.error_code.code}] {self.message}"
        
        # Add details if present
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            full_message += f" | {details_str}"
        
        # Log the error with full context
        logger.error(
            full_message,
            extra={
                "error_code": self.error_code.code,
                "details": self.details,
                "cause": str(self.cause) if self.cause else None,
            },
            exc_info=True
        )
        
        super().__init__(full_message)


class ConfigurationError(LangChainQAError):
    """Raised when there is a configuration error."""
    error_code = ErrorCode.CONFIGURATION_ERROR


class DocumentProcessingError(LangChainQAError):
    """Raised when there is an error processing a document."""
    error_code = ErrorCode.DOCUMENT_PROCESSING_ERROR


class VectorStoreError(LangChainQAError):
    """Raised when there is an error with the vector store."""
    error_code = ErrorCode.VECTOR_STORE_ERROR


class LLMError(LangChainQAError):
    """Raised when there is an error with the LLM."""
    error_code = ErrorCode.LLM_ERROR


class RetrievalError(LangChainQAError):
    """Raised when there is an error during document retrieval."""
    error_code = ErrorCode.RETRIEVAL_ERROR


class GenerationError(LangChainQAError):
    """Raised when there is an error during text generation."""
    error_code = ErrorCode.GENERATION_ERROR


class ValidationError(LangChainQAError, ValueError):
    """Raised when input validation fails."""
    error_code = ErrorCode.VALIDATION_ERROR


class AuthenticationError(LangChainQAError, PermissionError):
    """Raised when authentication or authorization fails."""
    error_code = ErrorCode.AUTHENTICATION_ERROR


class RateLimitError(LangChainQAError):
    """Raised when API rate limits are exceeded."""
    error_code = ErrorCode.RATE_LIMIT_ERROR


class ResourceNotFoundError(LangChainQAError, FileNotFoundError):
    """Raised when a requested resource is not found."""
    error_code = ErrorCode.RESOURCE_NOT_FOUND


class UnsupportedOperationError(LangChainQAError, NotImplementedError):
    """Raised when an operation is not supported."""
    error_code = ErrorCode.UNSUPPORTED_OPERATION


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
