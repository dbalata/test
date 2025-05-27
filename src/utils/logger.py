"""
Logging configuration for the LangChain Q&A System.

This module provides a centralized logging configuration that can be used throughout
the application. It supports both console and file logging with configurable log levels.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from ..config import settings


def setup_logger(
    name: str = "langchain_qa",
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return a logger with the specified settings.

    Args:
        name: Name of the logger
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to console.

    Returns:
        Configured logger instance
    """
    # Default log level
    default_log_level = "INFO"
    
    # Get log level from settings if not provided
    if log_level is None:
        try:
            log_level = getattr(settings, 'app', {}).get('log_level', default_log_level).upper()
        except Exception as e:
            print(f"Warning: Could not get log level from settings: {e}")
            log_level = default_log_level
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent adding handlers multiple times in case of module reload
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log file is specified
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "langchain_qa") -> logging.Logger:
    """
    Get a logger with the specified name.
    
    This is a convenience function that uses the default configuration.
    
    Args:
        name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Create default logger instance
logger = get_logger()

__all__ = ["setup_logger", "get_logger", "logger"]
