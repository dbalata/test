"""
Command-line interface for the LangChain application.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor
from src.qa_system import QASystem
from src.agents import create_research_agent, MultiAgentSystem
from src.code_generator import CodeGenerator
from src.utils import setup_logging, validate_environment, DocumentStats
from langchain.memory import ConversationBufferWindowMemory


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="LangChain Document Q&A System")
    parser.add_argument("--mode", choices=["web", "cli", "codegen"], default="web", help="Run mode")
    parser.add_argument("--docs", nargs="+", help="Document files to process")
    parser.add_argument("--url", help="URL to scrape and process")
    parser.add_argument("--query", help="Query to ask (CLI mode only)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Code generation arguments
    parser.add_argument("--template", help="Code template name (codegen mode)")
    parser.add_argument("--description", help="Code description for AI generation (codegen mode)")
    parser.add_argument("--language", default="python", help="Programming language (codegen mode)")
    parser.add_argument("--list-templates", action="store_true", help="List available code templates")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    logger = setup_logging("DEBUG" if args.verbose else "INFO")
    
    # Validate environment
    env_validation = validate_environment()
    if not env_validation.get("OPENAI_API_KEY", {}).get("valid"):
        logger.error("OPENAI_API_KEY is not set or invalid. Please set it in .env file.")
        sys.exit(1)
    
    if args.mode == "web":
        # Run Streamlit app
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"])
    
    elif args.mode == "codegen":
        # Code generation mode
        generator = CodeGenerator()
        
        if args.list_templates:
            print("Available Code Templates:")
            templates = generator.get_available_templates()
            for name, desc in templates.items():
                print(f"  • {name}: {desc}")
            return
        
        if args.template:
            print(f"Using template: {args.template}")
            print("Note: Template-based generation requires specific parameters.")
            print("Use the web interface for interactive template generation.")
            
        elif args.description:
            print(f"Generating {args.language} code for: {args.description}")
            try:
                result = generator.generate_with_ai(
                    description=args.description,
                    language=args.language
                )
                
                if result['success']:
                    print("\n" + "="*50)
                    print("GENERATED CODE")
                    print("="*50)
                    
                    result_data = result['result']
                    print(f"Explanation: {result_data['explanation']}")
                    
                    for i, block in enumerate(result_data['code_blocks'], 1):
                        print(f"\nCode Block {i} ({block['language']}):")
                        print("-" * 30)
                        print(block['code'])
                    
                    if result_data['dependencies']:
                        print(f"\nDependencies:")
                        for dep in result_data['dependencies']:
                            print(f"  • {dep}")
                            
                else:
                    print(f"Error generating code: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Code generation error: {e}")
        
        else:
            print("Please provide --description for AI generation or --template for template-based generation")
            print("Use --list-templates to see available templates")
    
    elif args.mode == "cli":
        # Run CLI mode
        if not (args.docs or args.url):
            logger.error("Please provide documents (--docs) or URL (--url) to process")
            sys.exit(1)
        
        # Process documents
        processor = DocumentProcessor()
        
        if args.docs:
            for doc_path in args.docs:
                if Path(doc_path).exists():
                    # For CLI, we need to handle file reading differently
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        processor.add_text(f.read(), source=doc_path)
                    logger.info(f"Added document: {doc_path}")
                else:
                    logger.warning(f"File not found: {doc_path}")
        
        if args.url:
            processor.add_url(args.url)
            logger.info(f"Added URL: {args.url}")
        
        # Create vector store
        vector_store = processor.create_vector_store()
        logger.info("Vector store created")
        
        # Initialize QA system
        memory = ConversationBufferWindowMemory(k=10, return_messages=True)
        qa_system = QASystem(vector_store=vector_store, memory=memory)
        
        # Document statistics
        doc_stats = DocumentStats.calculate_stats(processor.documents)
        logger.info(f"Processed {doc_stats['total_documents']} documents")
        logger.info(f"Total words: {doc_stats['total_words']}")
        
        if args.query:
            # Process single query
            result = qa_system.ask_question(args.query)
            print(f"\nQuestion: {args.query}")
            print(f"Answer: {result['answer']}")
            
            if result['sources']:
                print(f"\nSources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['metadata'].get('source', 'Unknown')}")
        
        else:
            # Interactive mode
            print("\nInteractive Q&A mode. Type 'quit' to exit.")
            
            while True:
                query = input("\nYour question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                try:
                    result = qa_system.ask_question(query)
                    print(f"\nAnswer: {result['answer']}")
                    
                    if args.verbose and result['sources']:
                        print(f"\nSources:")
                        for i, source in enumerate(result['sources'], 1):
                            print(f"{i}. {source['content'][:100]}...")
                
                except Exception as e:
                    logger.error(f"Error processing query: {e}")


if __name__ == "__main__":
    main()
