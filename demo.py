"""
Demonstration script showing various LangChain application features.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor
from src.qa_system import QASystem
from src.agents import create_research_agent, MultiAgentSystem
from src.utils import setup_logging, DocumentStats, PerformanceMonitor
from langchain.memory import ConversationBufferWindowMemory


def demo_document_processing():
    """Demonstrate document processing capabilities."""
    print("\n" + "="*50)
    print("DOCUMENT PROCESSING DEMO")
    print("="*50)
    
    processor = DocumentProcessor()
    
    # Add sample documents
    data_dir = Path("data")
    for file_path in data_dir.glob("*.txt"):
        print(f"Processing: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            processor.add_text(f.read(), source=str(file_path))
    
    # Get document info
    info = processor.get_document_info()
    print(f"\nDocument Info:")
    print(f"Total documents: {info['total_documents']}")
    print(f"Sources: {info['sources']}")
    print(f"Types: {info['types']}")
    
    # Create vector store
    print("\nCreating vector store...")
    vector_store = processor.create_vector_store()
    print(f"Vector store created with {len(vector_store.get()['ids'])} chunks")
    
    return vector_store


def demo_qa_system(vector_store):
    """Demonstrate Q&A system capabilities."""
    print("\n" + "="*50)
    print("Q&A SYSTEM DEMO")
    print("="*50)
    
    memory = ConversationBufferWindowMemory(k=5, return_messages=True)
    qa_system = QASystem(vector_store=vector_store, memory=memory)
    
    # Sample questions
    questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the main components of LangChain?",
        "What are some applications of AI in healthcare?",
        "How can I build a chatbot with LangChain?"
    ]
    
    performance_monitor = PerformanceMonitor()
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 40)
        
        try:
            import time
            start_time = time.time()
            
            result = qa_system.ask_question(question)
            
            response_time = time.time() - start_time
            performance_monitor.record_query(response_time, success=True)
            
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Sources: {len(result['sources'])} documents")
            print(f"Response time: {response_time:.2f}s")
            
        except Exception as e:
            performance_monitor.record_query(0, success=False)
            print(f"Error: {str(e)}")
    
    # Performance stats
    print(f"\nPerformance Statistics:")
    stats = performance_monitor.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return qa_system


def demo_advanced_features(qa_system):
    """Demonstrate advanced features."""
    print("\n" + "="*50)
    print("ADVANCED FEATURES DEMO")
    print("="*50)
    
    # Document summarization
    print("\nGenerating document summary...")
    try:
        summary = qa_system.summarize_documents()
        print(f"Summary: {summary[:200]}...")
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    # Similar document search
    print("\nFinding similar documents...")
    try:
        similar_docs = qa_system.get_similar_documents("machine learning algorithms", k=3)
        print(f"Found {len(similar_docs)} similar documents")
        for i, doc in enumerate(similar_docs, 1):
            print(f"{i}. {doc.page_content[:100]}...")
    except Exception as e:
        print(f"Error finding similar documents: {e}")
    
    # Analysis mode
    print("\nTesting analysis mode...")
    try:
        result = qa_system.ask_question(
            "Compare machine learning and deep learning approaches",
            analysis_mode=True
        )
        print(f"Analysis: {result['answer'][:200]}...")
    except Exception as e:
        print(f"Error in analysis mode: {e}")


def demo_agents():
    """Demonstrate agent capabilities."""
    print("\n" + "="*50)
    print("AGENTS DEMO")
    print("="*50)
    
    try:
        # Create research agent
        print("Creating research agent...")
        research_agent = create_research_agent()
        
        # Test queries (Note: These require API keys to work fully)
        test_queries = [
            "What are the latest developments in AI?",
            "Calculate the compound interest for $1000 at 5% for 10 years",
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                # Note: This will fail without proper API keys, but demonstrates structure
                print("Agent would process this query with available tools")
                print("Tools available: web_search, python_calculator")
            except Exception as e:
                print(f"Agent error (expected without API keys): {str(e)[:100]}...")
    
    except Exception as e:
        print(f"Error creating agents: {e}")


def demo_text_processing():
    """Demonstrate text processing utilities."""
    print("\n" + "="*50)
    print("TEXT PROCESSING DEMO")
    print("="*50)
    
    from src.utils import TextProcessor
    
    sample_text = """
    Artificial intelligence is revolutionizing many industries. Machine learning algorithms
    can process vast amounts of data and identify patterns that humans might miss. Deep
    learning, a subset of machine learning, uses neural networks with multiple layers to
    achieve remarkable results in image recognition, natural language processing, and more.
    """
    
    processor = TextProcessor()
    
    # Clean text
    cleaned = processor.clean_text(sample_text)
    print(f"Cleaned text: {cleaned[:100]}...")
    
    # Extract keywords
    keywords = processor.extract_keywords(sample_text)
    print(f"Keywords: {keywords}")
    
    # Summarize
    summary = processor.summarize_text(sample_text)
    print(f"Summary: {summary}")


def demo_document_stats():
    """Demonstrate document statistics."""
    print("\n" + "="*50)
    print("DOCUMENT STATISTICS DEMO")
    print("="*50)
    
    # Load sample documents
    documents = []
    data_dir = Path("data")
    
    for file_path in data_dir.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            from langchain.schema import Document
            doc = Document(
                page_content=f.read(),
                metadata={'source': str(file_path), 'type': 'text_file'}
            )
            documents.append(doc)
    
    # Calculate statistics
    stats = DocumentStats.calculate_stats(documents)
    
    print("Document Statistics:")
    for key, value in stats.items():
        if key not in ['sources', 'metadata_fields']:
            print(f"{key}: {value}")
    
    print(f"Sources: {', '.join(stats['sources'])}")
    print(f"Metadata fields: {', '.join(stats['metadata_fields'])}")


def demo_code_generation():
    """Demonstrate code generation capabilities."""
    print("\n" + "="*50)
    print("CODE GENERATION DEMO")
    print("="*50)
    
    try:
        from src.code_generator import CodeGenerator
        
        generator = CodeGenerator()
        
        # Show available templates
        print("Available Templates:")
        templates = generator.get_available_templates()
        for name, desc in templates.items():
            print(f"  • {name}: {desc}")
        
        print(f"\n{'-'*40}")
        print("Template Example: Python Class")
        print("-"*40)
        
        # Generate a Python class from template
        result = generator.generate_from_template(
            'python_class',
            class_name='DataProcessor',
            description='A class for processing and analyzing data',
            init_params=', data_source: str',
            init_body='self.data_source = data_source\n        self.data = []',
            methods='''def load_data(self):
        """Load data from source."""
        # Implementation here
        pass
    
    def process_data(self):
        """Process the loaded data."""
        # Implementation here
        return self.data'''
        )
        
        if result['success']:
            print("✅ Generated Python class:")
            print(result['code'])
        else:
            print(f"❌ Error: {result['error']}")
        
        # Demo template-based API generation
        print(f"\n{'-'*40}")
        print("Template Example: Flask API")
        print("-"*40)
        
        api_result = generator.generate_from_template(
            'flask_api',
            endpoint='users',
            method='GET',
            function_name='get_users',
            description='Get list of all users',
            function_body='''# Get users from database
    users = [
        {"id": 1, "name": "John Doe", "email": "john@example.com"},
        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
    ]''',
            return_value='{"users": users, "count": len(users)}'
        )
        
        if api_result['success']:
            print("✅ Generated Flask API endpoint:")
            print(api_result['code'][:300] + "...")
        else:
            print(f"❌ Error: {api_result['error']}")
            
        print(f"\n{'-'*40}")
        print("AI-Powered Generation Example")
        print("-"*40)
        
        # Only run AI generation if we have a valid API key
        if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != 'test-key-for-import-testing':
            ai_result = generator.generate_with_ai(
                description="Create a Python function that calculates the fibonacci sequence up to n terms using memoization for optimization",
                language="python"
            )
            
            if ai_result['success']:
                print("✅ AI-generated code:")
                result_data = ai_result['result']
                print(f"Explanation: {result_data['explanation'][:200]}...")
                if result_data['code_blocks']:
                    print(f"Code preview: {result_data['code_blocks'][0]['code'][:200]}...")
            else:
                print(f"❌ AI generation error: {ai_result['error']}")
        else:
            print("⚠️ Skipping AI generation (requires valid OpenAI API key)")
        
    except Exception as e:
        print(f"❌ Code generation demo error: {str(e)}")


def main():
    """Run all demonstrations."""
    print("LangChain Application Demonstration")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not found in environment.")
        print("Some features will not work without proper API keys.")
        print("Please set your API key in the .env file to test all features.\n")
    
    try:
        # Run demonstrations
        demo_text_processing()
        demo_document_stats()
        
        # These require API keys
        if os.getenv("OPENAI_API_KEY"):
            vector_store = demo_document_processing()
            qa_system = demo_qa_system(vector_store)
            demo_advanced_features(qa_system)
            demo_agents()
            demo_code_generation()
        else:
            print("\nSkipping API-dependent demos (no OpenAI API key)")
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\nTo run the full application:")
        print("1. Set your OPENAI_API_KEY in .env file")
        print("2. Run: streamlit run app.py")
        print("3. Or use CLI: python cli.py --mode cli --docs data/ai_overview.txt")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\nDemo encountered an error: {e}")


if __name__ == "__main__":
    main()
