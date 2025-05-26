# Advanced LangChain Document Q&A Web Application

A comprehensive web-based LangChain application for document processing, retrieval-augmented generation (RAG), and intelligent question-answering with a modern web interface.

## Features

### 🔧 Core Capabilities
- **Multi-format Document Processing**: PDF, TXT, Markdown files, and web scraping
- **Vector Storage**: ChromaDB with OpenAI embeddings for semantic search
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
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here  # Optional, for enhanced web search
```

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

## License

This project is licensed under the MIT License. Please ensure compliance with the terms of service of any third-party APIs and services used in the application.

---

**Built with ❤️ using LangChain and modern AI technologies**