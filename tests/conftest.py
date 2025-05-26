"""
Pytest configuration and fixtures for testing the LangChain Q&A System.
"""
import os
import sys
from pathlib import Path
from typing import Generator, Any

import pytest
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
load_dotenv()

# Configure logging for tests
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_VECTOR_STORE_DIR = TEST_DATA_DIR / "vector_store"

# Create test data directory if it doesn't exist
TEST_DATA_DIR.mkdir(exist_ok=True)

# Fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    
    # Create a test .env file if it doesn't exist
    test_env = Path(__file__).parent.parent / ".env.test"
    if not test_env.exists():
        test_env.write_text(
            """# Test environment variables
            OPENAI_API_KEY=test_key
            OPENAI_API_BASE=http://test-api-base
            VECTOR_STORE_TYPE=chroma
            VECTOR_STORE_PERSIST_DIR=./data/test_vector_store
            """
        )
    
    # Load test environment
    load_dotenv(test_env)
    
    yield  # Run tests
    
    # Clean up after tests
    if TEST_VECTOR_STORE_DIR.exists():
        import shutil
        shutil.rmtree(TEST_VECTOR_STORE_DIR)

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.app.log_level = "INFO"
    mock.llm.api_key = "test_key"
    mock.llm.base_url = "http://test-api-base"
    mock.vector_store.type = "chroma"
    mock.vector_store.persist_directory = str(TEST_VECTOR_STORE_DIR)
    mock.vector_store.collection_name = "test_collection"
    mock.vector_store.embedding_model = "text-embedding-3-small"
    mock.vector_store.chunk_size = 1000
    mock.vector_store.chunk_overlap = 200
    
    return mock

@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 384]  # Mock embedding vector
    mock.embed_query.return_value = [0.1] * 384  # Mock embedding vector
    
    return mock

@pytest.fixture
def sample_documents():
    """Generate sample documents for testing."""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test1.txt", "type": "test"}
        ),
        Document(
            page_content="Machine learning is a subset of AI.",
            metadata={"source": "test2.txt", "type": "test"}
        ),
        Document(
            page_content="Neural networks are used in deep learning.",
            metadata={"source": "test3.txt", "type": "test"}
        ),
    ]

@pytest.fixture
def sample_text_file() -> Generator[str, None, None]:
    """Create a sample text file for testing."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""This is a test document.
        It has multiple lines.
        And some more content for testing.""")
        f.flush()
        yield f.name
    
    # Clean up
    if os.path.exists(f.name):
        os.unlink(f.name)
