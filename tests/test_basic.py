"""Basic tests to verify the test setup."""

def test_imports():
    """Test that required packages can be imported."""
    import langchain_core
    import pytest
    from src.document_processor import DocumentProcessor
    
    assert langchain_core is not None
    assert pytest is not None
    assert DocumentProcessor is not None

def test_testing_environment():
    """Test that the testing environment is properly set up."""
    import os
    assert os.environ.get("TESTING") == "true"

def test_temp_dir_fixture(tmp_path):
    """Test that the tmp_path fixture works."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    assert test_file.read_text() == "test"
    assert test_file.exists()
