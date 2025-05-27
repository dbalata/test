"""
Utility functions and helpers for the LangChain application.
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import os


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    
    return logger


class ConfigManager:
    """Manage application configuration."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "embedding_model": "text-embedding-ada-002",
            "llm_model": "gpt-3.5-turbo",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_tokens": 1000,
            "temperature": 0.1,
            "vector_store_path": "./chroma_db",
            "retrieval_k": 5
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
        
        return default_config
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value


class DocumentStats:
    """Calculate and display document statistics."""
    
    @staticmethod
    def calculate_stats(documents: List) -> Dict[str, Any]:
        """Calculate statistics for a list of documents."""
        if not documents:
            return {"error": "No documents provided"}
        
        stats = {
            "total_documents": len(documents),
            "total_characters": 0,
            "total_words": 0,
            "average_length": 0,
            "sources": set(),
            "document_types": {},
            "metadata_fields": set()
        }
        
        for doc in documents:
            # Content statistics
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            stats["total_characters"] += len(content)
            stats["total_words"] += len(content.split())
            
            # Metadata analysis
            if hasattr(doc, 'metadata') and doc.metadata:
                for key in doc.metadata:
                    stats["metadata_fields"].add(key)
                
                source = doc.metadata.get('source', 'Unknown')
                doc_type = doc.metadata.get('type', 'Unknown')
                
                stats["sources"].add(source)
                stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
        
        # Calculate averages
        if stats["total_documents"] > 0:
            stats["average_length"] = stats["total_characters"] / stats["total_documents"]
            stats["average_words"] = stats["total_words"] / stats["total_documents"]
        
        # Convert sets to lists for JSON serialization
        stats["sources"] = list(stats["sources"])
        stats["metadata_fields"] = list(stats["metadata_fields"])
        
        return stats


class TextProcessor:
    """Text processing utilities."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Strip and normalize
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using simple frequency analysis."""
        import re
        from collections import Counter
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 
            'was', 'one', 'our', 'had', 'has', 'what', 'were', 'said', 'each', 
            'which', 'she', 'their', 'time', 'will', 'about', 'this', 'that',
            'from', 'they', 'been', 'have', 'more', 'when', 'who', 'would'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get most common words
        word_freq = Counter(filtered_words)
        keywords = [word for word, _ in word_freq.most_common(max_keywords)]
        
        return keywords
    
    @staticmethod
    def summarize_text(text: str, max_sentences: int = 3) -> str:
        """Create a simple extractive summary."""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + '.'
        
        # Simple scoring based on sentence length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split())  # Word count
            if i < len(sentences) * 0.3:  # Early sentences get bonus
                score *= 1.2
            scored_sentences.append((score, sentence))
        
        # Get top sentences
        scored_sentences.sort(reverse=True)
        top_sentences = [sent for _, sent in scored_sentences[:max_sentences]]
        
        return '. '.join(top_sentences) + '.'


class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "queries_processed": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "errors": 0,
            "successful_queries": 0
        }
        self.start_time = datetime.now()
    
    def record_query(self, response_time: float, success: bool = True):
        """Record a query execution."""
        self.metrics["queries_processed"] += 1
        
        if success:
            self.metrics["successful_queries"] += 1
            self.metrics["total_response_time"] += response_time
            self.metrics["average_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["successful_queries"]
            )
        else:
            self.metrics["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        uptime = datetime.now() - self.start_time
        
        return {
            **self.metrics,
            "uptime_seconds": uptime.total_seconds(),
            "success_rate": (
                self.metrics["successful_queries"] / max(self.metrics["queries_processed"], 1)
            ) * 100,
            "queries_per_minute": (
                self.metrics["queries_processed"] / max(uptime.total_seconds() / 60, 1)
            )
        }


def validate_environment() -> Dict[str, bool]:
    """Validate that required environment variables are set."""
    required_vars = [
        "LLM_API_KEY"  # OpenRouter API key
    ]
    
    optional_vars = [
        "SERPAPI_API_KEY",
        "LLM_BASE_URL",
        "LLM_PROVIDER",
        "LLM_MODEL"
    ]
    
    validation = {}
    
    for var in required_vars:
        validation[var] = {
            "required": True,
            "present": bool(os.getenv(var)),
            "valid": bool(os.getenv(var) and len(os.getenv(var)) > 10)
        }
    
    for var in optional_vars:
        validation[var] = {
            "required": False,
            "present": bool(os.getenv(var)),
            "valid": True  # These are optional, so always valid if present
        }
    
    return validation


def format_sources(sources: List[Dict]) -> str:
    """Format source documents for display."""
    if not sources:
        return "No sources available."
    
    formatted = []
    for i, source in enumerate(sources, 1):
        content = source.get('content', 'No content')
        metadata = source.get('metadata', {})
        
        source_type = metadata.get('type', 'Unknown')
        source_name = metadata.get('source', 'Unknown')
        
        formatted.append(f"**Source {i}** ({source_type}): {source_name}")
        formatted.append(f"Content: {content}")
        formatted.append("---")
    
    return "\n".join(formatted)
