# Advanced LangChain Document Q&A System

A comprehensive LangChain application demonstrating advanced features including document processing, retrieval-augmented generation (RAG), intelligent agents, and a web interface.

## Features

### 🔧 Core Capabilities
- **Multi-format Document Processing**: PDF, TXT, Markdown files, and web scraping
- **Vector Storage**: ChromaDB with OpenAI embeddings for semantic search
- **Retrieval-Augmented Generation (RAG)**: Contextual Q&A with source citations
- **Conversation Memory**: Maintains context across multiple interactions
- **Web Interface**: Beautiful Streamlit-based UI

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
- **CLI Interface**: Command-line interface for automation

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd /workspaces/test

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here  # Optional, for enhanced web search
```

### 3. Run the Application

#### Web Interface (Recommended)
```bash
streamlit run app.py
```

#### Command Line Interface
```bash
# Process documents and start interactive Q&A
python cli.py --mode cli --docs data/sample.txt

# Process a URL
python cli.py --mode cli --url https://example.com

# Ask a single question
python cli.py --mode cli --docs data/sample.txt --query "What is the main topic?"
```

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

#### 1. Academic Research
```python
# Upload research papers (PDFs)
# Ask: "What are the main methodological differences between these studies?"
# Use sentiment analysis to understand author perspectives
```

#### 2. Business Intelligence
```python
# Upload reports and web scrape company pages
# Ask: "What are the key growth drivers mentioned?"
# Use topic extraction to identify strategic themes
```

#### 3. Technical Documentation
```python
# Upload code documentation and tutorials
# Ask: "How do I implement authentication?"
# Use code analysis for technical insights
```

## Architecture

### Core Components

1. **DocumentProcessor** (`src/document_processor.py`)
   - Handles multiple document formats
   - Creates embeddings and vector storage
   - Manages document metadata

2. **QASystem** (`src/qa_system.py`)
   - Implements RAG pipeline
   - Manages conversation memory
   - Provides confidence scoring

3. **Agents** (`src/agents.py`)
   - Research agent with web search
   - Document analysis tools
   - Multi-agent coordination

4. **Utils** (`src/utils.py`)
   - Configuration management
   - Performance monitoring
   - Text processing utilities

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
Enable verbose logging for troubleshooting:

```bash
python cli.py --verbose --mode cli --docs your_document.pdf
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

This is a demonstration project showcasing LangChain capabilities. Feel free to:

1. Add new document processors
2. Implement additional agents
3. Enhance the UI/UX
4. Add new analysis tools
5. Improve performance optimizations

## Dependencies

- **LangChain**: Core framework for LLM applications
- **OpenAI**: Language model and embeddings
- **ChromaDB**: Vector database for semantic search
- **Streamlit**: Web interface framework
- **BeautifulSoup**: Web scraping capabilities
- **PyPDF2**: PDF document processing

## License

This project is for educational and demonstration purposes. Please ensure compliance with API terms of service and applicable licenses for production use.

---

**Built with ❤️ using LangChain and modern AI technologies**