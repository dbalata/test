"""
Test suite for the LangChain Q&A System.

This package contains all the test cases for the application.
"""
from pathlib import Path

# Define test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"

# Create test data directory if it doesn't exist
TEST_DATA_DIR.mkdir(exist_ok=True)

__all__ = ["TEST_DATA_DIR"]
