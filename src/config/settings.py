"""Application configuration settings using Pydantic.

This module provides a centralized configuration management system using Pydantic's
BaseSettings. It loads configuration from environment variables with support for
.env files.
"""
from enum import Enum
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union

from pydantic import Field, field_validator, model_validator, HttpUrl, field_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_core import Url
import warnings
import os

from src.utils.exceptions import ConfigurationError, ErrorCode

# Suppress deprecation warnings from pydantic
warnings.filterwarnings("ignore", message="pydantic settings")

# Environment variable prefixes
ENV_PREFIX = "LANGCHAIN_QA_"

# Constants
DEFAULT_VECTOR_STORE_DIR = "./data/vector_store"


class AppSettings(BaseSettings):
    """Application settings.
    
    Attributes:
        name: Name of the application
        version: Version string (semver)
        debug: Enable debug mode
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        docs_url: URL path for API documentation
        openapi_url: URL path for OpenAPI schema
    """
    name: str = Field(
        "langchain-qa-system",
        min_length=1,
        description="Name of the application"
    )
    version: str = Field(
        "0.1.0",
        pattern=r'^\d+\.\d+\.\d+$',
        description="Version string (semver)"
    )
    debug: bool = Field(
        False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        "INFO",
        pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$',
        description="Logging level"
    )
    docs_url: str = Field(
        "/docs",
        description="URL path for API documentation"
    )
    openapi_url: str = Field(
        "/openapi.json",
        description="URL path for OpenAPI schema"
    )
    
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}APP_", extra="forbid")


class LLMSettings(BaseSettings):
    """Settings for the LLM provider.
    
    Attributes:
        provider: LLM provider to use (e.g., 'openrouter', 'openai', 'anthropic')
        model: Model name with provider prefix (e.g., 'openai/gpt-4-turbo')
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate (1-8192)
        top_p: Nucleus sampling parameter (0-1)
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        api_key: API key for the provider
        base_url: Base URL for the API
    """
    provider: str = Field(
        "openrouter",
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="LLM provider identifier"
    )
    model: str = Field(
        "openai/gpt-4-turbo-preview",
        min_length=1,
        description="Model identifier with provider prefix"
    )
    temperature: float = Field(
        0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2)"
    )
    max_tokens: int = Field(
        2048,
        ge=1,
        le=8192,
        description="Maximum tokens to generate"
    )
    top_p: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    frequency_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    presence_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    api_key: Optional[str] = Field(
        None,
        min_length=1,
        description="API key for the LLM provider"
    )
    base_url: str = Field(
        "https://openrouter.ai/api/v1",
        description="Base URL for the LLM API"
    )
    
    model_config = SettingsConfigDict(
        env_prefix=f"{ENV_PREFIX}LLM_",
        extra="ignore",  # Changed from 'forbid' to 'ignore' to allow extra env vars
        validate_default=True,
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    @field_serializer('base_url')
    def serialize_base_url(self, url: Optional[Union[str, HttpUrl]]) -> Optional[str]:
        if url is None:
            return None
        return str(url)
    
    @field_validator('base_url', mode='before')
    @classmethod
    def validate_base_url(cls, v):
        if v is None:
            return "https://openrouter.ai/api/v1"
        if isinstance(v, HttpUrl):
            return str(v)
        return v
        
    @field_validator('api_key', mode='before')
    @classmethod
    def validate_api_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate API key is provided for non-local providers."""
        if os.environ.get("TESTING") == "true":
            return v or "test_key"
            
        # Try to get the provider from info.data or use 'local' as default
        provider = info.data.get("provider", "local") if hasattr(info, "data") else "local"
        
        # If no API key was provided, try to get it from environment variables directly
        if not v:
            v = os.environ.get(f"{ENV_PREFIX}LLM_API_KEY") or os.environ.get("LLM_API_KEY")
            
        if not v and provider != "local":
            raise ConfigurationError(
                f"API key is required for provider: {provider}",
                details={
                    "provider": provider,
                    "env_vars_checked": [f"{ENV_PREFIX}LLM_API_KEY", "LLM_API_KEY"]
                },
                error_code=ErrorCode.CONFIGURATION_ERROR
            )
            
        return v


class VectorStoreSettings(BaseSettings):
    """Vector store configuration.
    
    Attributes:
        type: Type of vector store ('chroma', 'faiss', 'weaviate')
        persist_directory: Directory to store vector data
        collection_name: Name of the collection/index
        embedding_model: Model for generating embeddings
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks
        search_kwargs: Additional search parameters
    """
    type: str = Field(
        "chroma",
        pattern=r'^(chroma|faiss|weaviate|pinecone|qdrant)$',
        description="Type of vector store"
    )
    persist_directory: Path = Field(
        default_factory=lambda: Path(DEFAULT_VECTOR_STORE_DIR),
        description="Directory to store vector data"
    )
    collection_name: str = Field(
        "documents",
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Name of the collection/index"
    )
    embedding_model: str = Field(
        "text-embedding-3-small",
        description="Model for generating embeddings"
    )
    chunk_size: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Size of text chunks"
    )
    chunk_overlap: int = Field(
        200,
        ge=0,
        le=1000,
        description="Overlap between chunks"
    )
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"k": 5},
        description="Additional search parameters"
    )
    
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}VECTOR_STORE_", extra="forbid")
    
    @field_validator("persist_directory")
    @classmethod
    def validate_persist_directory(cls, v: Path) -> Path:
        """Ensure persist directory exists and is writable."""
        try:
            v.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = v / ".write_test"
            test_file.touch(exist_ok=True)
            test_file.unlink(missing_ok=True)
            return v.absolute()
        except (OSError, PermissionError) as e:
            raise ConfigurationError(
                f"Cannot access or create directory: {v}",
                details={"path": str(v)},
                error_code=ErrorCode.CONFIGURATION_ERROR
            ) from e


class MemorySettings(BaseSettings):
    """Memory and conversation settings."""

    window_size: int = 10
    memory_key: str = "chat_history"
    return_messages: bool = True
    output_key: str = "answer"
    input_key: str = "question"


class DocumentProcessorSettings(BaseSettings):
    """Document processing settings."""
    pass


class SearchSettings(BaseSettings):
    """Search settings."""
    pass


class LoggingSettings(BaseSettings):
    """Logging settings."""
    pass


class Settings(BaseSettings):
    """Application settings container.
    
    This class serves as the root configuration object that composes all
    other settings classes. It handles environment variable loading, validation,
    and provides helper methods for common configuration needs.
    
    Attributes:
        app: General application settings
        llm: LLM provider settings
        document_processor: Document processing settings
        vector_store: Vector store configuration
        search: Search-related settings
        logging: Logging configuration
    """
    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    document_processor: DocumentProcessorSettings = Field(
        default_factory=DocumentProcessorSettings
    )
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Changed from 'forbid' to 'ignore' to allow extra env vars
        validate_default=True,
        case_sensitive=True,  # Make case-sensitive to match environment variables exactly
        env_prefix=ENV_PREFIX,  # Add the environment prefix
    )
    
    def __init__(self, **data):
        """Initialize settings with proper validation and test environment handling."""
        # Handle test environment
        if os.environ.get("TESTING") == "true":
            data.setdefault("llm", {}).setdefault("api_key", "test_key")
            data.setdefault("vector_store", {}).setdefault(
                "persist_directory", 
                Path("./tests/data/vector_store").absolute()
            )
        
        # Ensure LLM settings get the API key from environment if not provided
        llm_data = data.get("llm", {})
        if not llm_data.get("api_key"):
            llm_data["api_key"] = os.environ.get(f"{ENV_PREFIX}LLM_API_KEY") or os.environ.get("LLM_API_KEY")
            data["llm"] = llm_data
        
        try:
            super().__init__(**data)
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialize settings",
                details={
                    "error": str(e),
                    "env_vars": {
                        "LLM_API_KEY": "***" if os.environ.get("LLM_API_KEY") else None,
                        f"{ENV_PREFIX}LLM_API_KEY": "***" if os.environ.get(f"{ENV_PREFIX}LLM_API_KEY") else None
                    }
                },
                error_code=ErrorCode.CONFIGURATION_ERROR
            ) from e

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.app.debug
        
    @property
    def is_testing(self) -> bool:
        """Check if running in test mode."""
        return os.environ.get("TESTING") == "true"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app.debug and not self.testing


def get_settings() -> Settings:
    """Create and return a new Settings instance.
    
    This function should be used to get a settings instance instead of 
    importing the settings object directly, to ensure proper initialization.
    """
    return Settings()

# For backward compatibility, create a default settings instance
# but note that this will raise an error if required environment variables are not set
settings = get_settings()

__all__ = ["settings", "get_settings"]
