"""
Application configuration settings using Pydantic.

This module provides a centralized configuration management system using Pydantic's
BaseSettings. It loads configuration from environment variables with support for
.env files.
"""
from enum import Enum
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, ClassVar, Literal, Type, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import warnings
import os
from typing import Optional, List, Dict, Any, ClassVar, Literal, Type, Union

# Suppress deprecation warnings from pydantic
warnings.filterwarnings("ignore", message="pydantic settings")

# Environment variable prefixes
ENV_PREFIX = "LANGCHAIN_QA_"


class AppSettings(BaseSettings):
    """Application settings."""

    name: str = "langchain-qa-system"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"


class LLMSettings(BaseSettings):
    """Settings for the LLM provider."""
    
    # Default provider
    provider: str = Field("openai", description="LLM provider to use (e.g., 'openai', 'anthropic')")
    model: str = Field("gpt-4-turbo-preview", description="Model name to use")
    temperature: float = Field(0.2, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Maximum number of tokens to generate")
    top_p: float = Field(1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, description="Presence penalty")
    api_key: Optional[str] = Field(None, description="API key for the LLM provider")
    base_url: Optional[str] = Field(None, description="Base URL for the LLM API")
    
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}LLM_", extra="ignore")
    
    @field_validator('api_key', mode='before')
    @classmethod
    def validate_api_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate API key is provided for non-local providers."""
        # Skip validation in test environment
        if os.environ.get("TESTING") == "true":
            return v or "test_key"
            
        # Get the provider from the values if available
        provider = info.data.get("provider", "local") if hasattr(info, "data") else "local"
            
        if not v and provider != "local":
            warnings.warn(
                f"No API key provided for {provider}. "
                "Some features may not work without authentication."
            )
        return v


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    type: str = "chroma"  # or 'faiss', 'weaviate', etc.
    persist_directory: Path = Path("./data/vector_store")
    collection_name: str = "documents"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"k": 5})

    @field_validator("persist_directory")
    @classmethod
    def validate_persist_directory(cls, v: Path) -> Path:
        """Ensure persist directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v


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
    """Application settings."""

    # Application settings
    app: AppSettings = Field(default_factory=AppSettings)
    
    # LLM settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    
    # Document processing settings
    document_processor: DocumentProcessorSettings = Field(
        default_factory=DocumentProcessorSettings
    )
    
    # Vector store settings
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    
    # Search settings
    search: SearchSettings = Field(default_factory=SearchSettings)
    
    # Logging settings
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    def __init__(self, **data):
        """Initialize settings with proper validation."""
        # Handle test environment
        if os.environ.get("TESTING") == "true":
            data.setdefault("llm", {}).setdefault("api_key", "test_key")
        super().__init__(**data)

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not (self.app.debug or self.testing)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app.debug


# Create settings instance
settings = Settings()

# For backward compatibility
__all__ = ["settings"]
