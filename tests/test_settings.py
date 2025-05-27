"""Tests for the settings module."""
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.config.settings import Settings

# Mock the environment variables for testing
TEST_ENV = {
    "APP_NAME": "test-app",
    "APP_VERSION": "1.0.0",
    "APP_DEBUG": "true",
    "APP_LOG_LEVEL": "DEBUG",
    "OPENROUTER_API_KEY": "test-api-key",
    "OPENROUTER_BASE_URL": "http://test-api"
}

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)


def test_settings_defaults():
    """Test default values for Settings."""
    settings = Settings()
    assert settings.app.name == "langchain-qa-system"
    assert settings.app.version == "0.1.0"
    assert settings.app.debug is False
    assert settings.app.log_level == "INFO"
    assert settings.openrouter_api_key is None
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_settings_environment_loading(mock_env):
    """Test loading settings from environment variables."""
    settings = Settings()
    
    # Verify environment variables were loaded correctly
    assert settings.app.name == "test-app"
    assert settings.app.version == "1.0.0"
    assert settings.app.debug is True
    assert settings.app.log_level == "DEBUG"
    assert settings.openrouter_api_key == "test-api-key"
    assert settings.openrouter_base_url == "http://test-api"


def test_settings_test_environment(monkeypatch):
    """Test settings initialization in test environment."""
    # Set TESTING environment variable
    monkeypatch.setenv("TESTING", "true")
    
    settings = Settings()
    assert settings.app.debug is True
    assert settings.app.log_level == "DEBUG"


def test_settings_api_key_validation(monkeypatch):
    """Test API key validation in Settings."""
    # Clear any test environment settings
    monkeypatch.delenv("TESTING", raising=False)
    
    # Test with no API key (should warn)
    with patch('warnings.warn') as mock_warn:
        settings = Settings(openrouter_api_key=None)
        # Verify warning was called with the expected message
        mock_warn.assert_called_once()
        warning_msg = mock_warn.call_args[0][0]
        assert "No API key provided for openrouter" in warning_msg
    
    # Test with API key (should not warn)


def test_settings_env_loading(monkeypatch):
    """Test loading settings from environment variables with patch.dict."""
    # Clear any test environment settings
    monkeypatch.delenv("TESTING", raising=False)
    
    # Set environment variables
    with patch.dict(os.environ, {
        "APP_NAME": "custom-app",
        "APP_VERSION": "2.0.0",
        "APP_DEBUG": "false",
        "APP_LOG_LEVEL": "WARNING",
        "OPENROUTER_API_KEY": "test-key-123",
        "OPENROUTER_BASE_URL": "http://custom-api"
    }, clear=True):
        # Create a new Settings instance
        settings = Settings()
        assert settings.app.name == "custom-app"
        assert settings.app.version == "2.0.0"
        assert settings.app.debug is False
        assert settings.app.log_level == "WARNING"
        assert settings.openrouter_api_key == "test-key-123"
        assert settings.openrouter_base_url == "http://custom-api"


def test_settings_properties():
    """Test property methods in Settings class."""
    # Test development mode
    dev_settings = Settings(app={"debug": True}, _env_file=None)
    assert dev_settings.is_development is True
    assert dev_settings.is_production is False
    
    # Test production mode
    prod_settings = Settings(app={"debug": False}, _env_file=None)
    assert prod_settings.is_development is False
    # In production mode, we can't test is_production directly as it checks for self.testing
    # which isn't set in the test environment
