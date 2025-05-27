"""Tests for the logger utility."""
import logging
from pathlib import Path
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock, call

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the functions we want to test
from src.utils import setup_logger, get_logger, logger as default_logger

# Mock the settings module
@pytest.fixture(autouse=True)
def mock_settings():
    """Mock the settings module."""
    with patch('src.config.settings') as mock_settings:
        # Create a mock for the app attribute
        mock_settings.app = MagicMock()
        mock_settings.app.log_level = "INFO"
        yield mock_settings


def test_setup_logger_default(mock_settings):
    """Test setting up logger with default settings."""
    # Setup mocks
    with patch('logging.getLogger') as mock_get_logger, \
         patch('logging.StreamHandler') as mock_handler_cls, \
         patch('logging.Formatter') as mock_formatter_cls:
        
        # Create a real logger for testing
        test_logger = logging.getLogger("test_logger")
        test_logger.handlers = []  # Clear any existing handlers
        mock_get_logger.return_value = test_logger
        
        # Mock the handler and formatter
        mock_handler = MagicMock()
        mock_handler_cls.return_value = mock_handler
        mock_formatter = MagicMock()
        mock_formatter_cls.return_value = mock_formatter
        
        # Setup mock logger with name attribute
        test_logger = MagicMock()
        test_logger.name = "langchain_qa"
        test_logger.handlers = []
        test_logger.level = logging.INFO
        mock_get_logger.return_value = test_logger
        
        # Call the function
        logger = setup_logger()
        
        # Verify the logger name and level were set correctly
        mock_get_logger.assert_called_with("langchain_qa")
        test_logger.setLevel.assert_called_once_with("INFO")
        
        # Verify the handler was created with the correct formatter
        mock_formatter_cls.assert_called_once_with(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        mock_handler.setFormatter.assert_called_once_with(mock_formatter)
        
        # Verify the handler was added to the logger
        test_logger.addHandler.assert_called_once_with(mock_handler)


def test_setup_logger_with_file():
    """Test setting up logger with file output."""
    # Setup mocks
    with patch('logging.getLogger') as mock_get_logger, \
         patch('logging.FileHandler') as mock_file_handler_cls, \
         patch('logging.StreamHandler') as mock_stream_handler_cls, \
         patch('logging.Formatter') as mock_formatter_cls:
        
        # Create a real logger for testing
        test_logger = logging.getLogger("test_file_logger")
        test_logger.handlers = []  # Clear any existing handlers
        mock_get_logger.return_value = test_logger
        
        # Mock the handlers and formatter
        mock_file_handler = MagicMock()
        mock_stream_handler = MagicMock()
        mock_file_handler_cls.return_value = mock_file_handler
        mock_stream_handler_cls.return_value = mock_stream_handler
        mock_formatter = MagicMock()
        mock_formatter_cls.return_value = mock_formatter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Setup mock logger with name attribute
            test_logger = MagicMock()
            test_logger.name = "test_logger"
            test_logger.handlers = []
            test_logger.level = logging.DEBUG
            mock_get_logger.return_value = test_logger
            
            # Call the function
            logger = setup_logger(
                name="test_logger",
                log_level="DEBUG",
                log_file=log_file
            )
            
            # Verify logger name and level were set correctly
            mock_get_logger.assert_called_with("test_logger")
            test_logger.setLevel.assert_called_once_with("DEBUG")
            
            # Verify the formatter was created with the correct format
            mock_formatter_cls.assert_called_with(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            
            # Verify the file handler was created with the correct path and encoding
            mock_file_handler_cls.assert_called_once_with(log_file, encoding='utf-8')
            
            # Verify both handlers were added to the logger
            assert test_logger.addHandler.call_count == 2
            test_logger.addHandler.assert_any_call(mock_stream_handler)
            test_logger.addHandler.assert_any_call(mock_file_handler)


def test_get_logger():
    """Test getting a logger with default configuration."""
    # Setup
    logger_name = "test_get_logger"
    
    # Create a real logger for testing
    test_logger = logging.getLogger(logger_name)
    test_logger.handlers = []  # Clear any existing handlers
    
    # Patch the getLogger function
    with patch('logging.getLogger') as mock_get_logger:
        # Setup mock logger with name attribute
        test_logger = MagicMock()
        test_logger.name = logger_name
        test_logger.handlers = []
        test_logger.level = 0  # NOTSET by default
        mock_get_logger.side_effect = [test_logger, test_logger]
        
        # Call the function
        logger = get_logger(logger_name)
        
        # Verify the logger was created with the correct name
        mock_get_logger.assert_called_with(logger_name)
        
        # Verify the logger is the same when called again with the same name
        assert get_logger(logger_name) is logger
        
        # Verify the logger has the expected properties
        mock_get_logger.assert_called_with(logger_name)
        assert test_logger is logger


def test_logger_levels():
    """Test that different log levels work as expected."""
    # Test with each log level
    test_cases = [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ]
    
    for level_name, level_value in test_cases:
        with patch('logging.getLogger') as mock_get_logger, \
             patch('logging.StreamHandler') as mock_handler_cls, \
             patch('logging.Formatter') as mock_formatter_cls:
            
            # Create a real logger for testing
            test_logger = logging.getLogger(f"test_{level_name.lower()}")
            test_logger.handlers = []  # Clear any existing handlers
            mock_get_logger.return_value = test_logger
            
            # Mock the handler and formatter
            mock_handler = MagicMock()
            mock_handler_cls.return_value = mock_handler
            mock_formatter = MagicMock()
            mock_formatter_cls.return_value = mock_formatter
            
            # Setup mock logger with name attribute
            test_logger = MagicMock()
            test_logger.name = f"test_{level_name.lower()}"
            test_logger.handlers = []
            test_logger.level = level_value
            mock_get_logger.side_effect = [test_logger]
            
            # Call setup_logger with the current log level
            logger = setup_logger(name=f"test_{level_name.lower()}", log_level=level_name)
            
            # Verify the logger's level was set correctly
            test_logger.setLevel.assert_called_once_with(level_name)
            
            # Verify the logger was created with the correct name and level
            mock_get_logger.assert_called_with(f"test_{level_name.lower()}")
            test_logger.setLevel.assert_called_once_with(level_name)


def test_logger_handlers_not_duplicated():
    """Test that handlers are not added multiple times."""
    # Use a unique logger name for this test
    logger_name = "test_no_duplicates"
    
    # Clear any existing logger with this name
    if logger_name in logging.Logger.manager.loggerDict:
        del logging.Logger.manager.loggerDict[logger_name]
    
    # Setup mocks
    with patch('logging.StreamHandler') as mock_handler_cls, \
         patch('logging.Formatter') as mock_formatter_cls:
        
        # Mock the handler and formatter
        mock_handler = MagicMock()
        mock_handler_cls.return_value = mock_handler
        mock_formatter = MagicMock()
        mock_formatter_cls.return_value = mock_formatter
        
        # First call to setup_logger - should add handlers
        logger1 = setup_logger(name=logger_name)
        
        # Get the number of handlers after first setup
        initial_handler_count = len(logging.getLogger(logger_name).handlers)
        
        # Second call to setup_logger with the same name
        logger2 = setup_logger(name=logger_name)
        
        # Should return the same logger instance
        assert logger1 is logger2
        
        # Should not have added any new handlers
        assert len(logging.getLogger(logger_name).handlers) == initial_handler_count
        
        # Clean up by removing the logger from the manager
        if logger_name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[logger_name]


class TestLoggerIntegration:
    """Integration tests for the logger."""
    
    def test_logger_output(self, capsys):
        """Test that log messages are properly formatted and output."""
        logger = setup_logger("test_output", "INFO")
        test_message = "Test log message"
        
        logger.info(test_message)
        captured = capsys.readouterr()
        
        assert test_message in captured.out
        assert "INFO" in captured.out
        assert "test_output" in captured.out
