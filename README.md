# Advanced LangChain Document Q&A Web Application

A comprehensive web-based LangChain application for document processing, retrieval-augmented generation (RAG), and intelligent question-answering with a modern web interface. Powered by OpenRouter for LLM access.

## Features

### 🔧 Core Capabilities
- **Multi-format Document Processing**: PDF, TXT, Markdown files, and web scraping
- **Vector Storage**: ChromaDB with embeddings for semantic search
- **Retrieval-Augmented Generation (RAG)**: Contextual Q&A with source citations
- **Conversation Memory**: Maintains context across multiple interactions
- **Modern Web Interface**: Beautiful Streamlit-based UI

### 🤖 Advanced AI Features
- **Intelligent Agents**: Research agent with web search and analysis tools
- **Multi-Agent System**: Routing queries to specialized agents
- **Sentiment Analysis**: Analyze document sentiment and emotional indicators
- **Topic Extraction**: Identify key themes and concepts
- **Code Analysis**: Analyze code snippets found in documents

### 🛠️ Technical Features
- **Configurable Settings**: JSON-based configuration management
- **Performance Monitoring**: Track response times and success rates
- **Comprehensive Logging**: Detailed logging with file and console output
- **Error Handling**: Robust error handling and validation

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root with your API keys:

```
# Required
LLM_API_KEY=your_openrouter_api_key_here

# Optional - override defaults if needed
# LLM_BASE_URL=https://openrouter.ai/api/v1
# LLM_PROVIDER=openrouter
# LLM_MODEL=openai/gpt-4-turbo-preview

# Optional - for enhanced web search
# SERPAPI_API_KEY=your_serpapi_key_here
```

### 3. Get an OpenRouter API Key

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Get your API key from the [keys page](https://openrouter.ai/keys)
3. Add it to your `.env` file as `LLM_API_KEY`

### 3. Run the Web Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage Guide

### Document Processing

The system supports multiple document sources:

1. **File Upload**: PDF, TXT, and Markdown files
2. **Web Scraping**: Extract content from URLs
3. **Manual Text**: Direct text input

### Question Types

The system excels at various question types:

- **Factual Questions**: "What is the main conclusion?"
- **Analytical Questions**: "What are the key themes?"
- **Comparative Questions**: "How do these approaches differ?"
- **Sentiment Questions**: "What's the overall tone?"

### Advanced Features

#### Research Agent
Enable the research agent for questions requiring web search:
- Current events and recent information
- Cross-referencing with external sources
- Real-time data and statistics

#### Multi-Agent Routing
The system automatically routes queries to appropriate agents:
- Document analysis for content-based questions
- Web search for current information needs
- Code analysis for technical content

## Running Tests

The project includes a comprehensive test suite to ensure code quality and reliability. To run the tests:

### Prerequisites

- Python 3.9+
- All dependencies from `requirements.txt`
- Test dependencies (install with `pip install -e ".[test]"`)

### Running All Tests

```bash
pytest
```

### Running Specific Test Files

```bash
# Run logger tests
pytest tests/test_logger.py -v

# Run exception handling tests
pytest tests/test_exceptions.py tests/test_exceptions_simple.py tests/test_exception_contract.py tests/test_actual_exceptions.py -v

# Run settings tests
pytest tests/test_settings.py -v
```

### Running with Coverage

To generate a coverage report:

```bash
pytest --cov=src --cov-report=term-missing
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Exception Tests**: Verify proper error handling

### Writing New Tests

When adding new features, please include corresponding tests. Follow these guidelines:

1. Place test files in the `tests/` directory
2. Name test files with `test_` prefix
3. Use descriptive test function names starting with `test_`
4. Include docstrings explaining what each test verifies

### Mocking

Use the `unittest.mock` library to mock external dependencies in tests. Common mocks include:

- API clients
- File I/O operations
- External services

Example:

```python
from unittest.mock import patch, MagicMock

@patch('module.ClassName')
def test_something(mock_class):
    mock_instance = mock_class.return_value
    mock_instance.method.return_value = "mocked response"
    # Test code here


### Example Workflows

#### Academic Research
- Upload research papers
- Ask: "What are the main methodological differences between these studies?"
- Use sentiment analysis to understand author perspectives

#### Business Intelligence
- Upload reports and web scrape company pages
- Ask: "What are the key growth drivers mentioned?"

#### Technical Documentation
- Upload API documentation and code examples
- Ask: "How do I implement OAuth authentication?"
- Find relevant code examples and implementation details

## Architecture

### Core Components

1. **DocumentProcessor**: Handles document loading, preprocessing, and vector storage
2. **QASystem**: Implements the RAG pipeline for question-answering
3. **Agents**: Specialized agents for different types of queries
4. **Utils**: Supporting utilities for configuration, logging, and text processing

### Data Flow

```
Documents → Processing → Vector Store → Retrieval → LLM → Response
                                    ↓
                               Conversation Memory
                                    ↓
                                Web Search (if needed)
```

## Configuration

Customize the system via `config.json`:

```json
{
  "embedding_model": "text-embedding-ada-002",
  "llm_model": "gpt-3.5-turbo",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "temperature": 0.1,
  "retrieval_k": 5
}
```

## Performance

### Optimization Features
- **Efficient Chunking**: Optimized text splitting for better retrieval
- **Caching**: Vector store persistence for faster startup
- **Memory Management**: Configurable conversation window
- **Batch Processing**: Handle multiple documents efficiently

### Monitoring
- Response time tracking
- Success rate monitoring
- Error logging and analysis
- Performance metrics dashboard

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure OPENAI_API_KEY is set in `.env`
   - Check API key validity and credits

2. **Document Processing Errors**
   - Verify file formats are supported
   - Check file encoding (UTF-8 recommended)
   - Ensure sufficient disk space for vector store

3. **Memory Issues**
   - Reduce chunk size for large documents
   - Adjust memory window size
   - Clear conversation history if needed

### Debug Mode
Enable verbose logging for troubleshooting by setting the environment variable before starting the application:

```bash
export LOG_LEVEL=DEBUG
streamlit run app.py
```

## Advanced Customization

### Adding New Document Types
Extend `DocumentProcessor` to support additional formats:

```python
def add_custom_format(self, file_path):
    # Custom processing logic
    pass
```

### Custom Agents
Create specialized agents for specific domains:

```python
def create_custom_agent():
    # Define tools and capabilities
    pass
```

### Enhanced Prompts
Modify prompts in `QASystem` for domain-specific responses:

```python
def get_domain_specific_prompt():
    # Custom prompt template
    pass
```

## Contributing

Contributions are welcome! Here are some ways you can contribute:

1. Add support for new document formats
2. Enhance the web interface with new features
3. Improve the document processing pipeline
4. Add new analysis and visualization tools
5. Optimize performance and scalability

## Dependencies

- **LangChain**: Core framework for LLM applications
- **OpenAI**: Language model and embeddings
- **ChromaDB**: Vector database for semantic search
- **Streamlit**: Web interface framework
- **BeautifulSoup**: Web scraping capabilities
- **PyPDF2**: PDF document processing

---

**Built with ❤️ using LangChain and modern AI technologies**