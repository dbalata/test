"""
Configuration package for the LangChain Q&A System.

This package provides a centralized configuration system using Pydantic settings.
It loads configuration from environment variables with support for .env files.
"""
from typing import TYPE_CHECKING

from .settings import settings as settings

if TYPE_CHECKING:
    from .settings import Settings

__all__ = ["settings"]
