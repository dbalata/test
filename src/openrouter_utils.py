"""
Utilities for working with OpenRouter API.
"""

import os
from typing import Dict, Any, Optional, List, Union
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


def get_openrouter_llm(
    model_name: str = "openai/gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> ChatOpenAI:
    """
    Initialize a ChatOpenAI instance configured for OpenRouter.
    
    Args:
        model_name: The model to use (e.g., 'openai/gpt-3.5-turbo')
        temperature: The temperature to use for sampling
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional arguments to pass to ChatOpenAI
        
    Returns:
        Configured ChatOpenAI instance
    """
    # Get the API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    # Configure default headers
    default_headers = {
        "HTTP-Referer": "http://localhost:8501",  # Your site URL
        "X-Title": "LangChain App",  # Your app name
    }
    
    # Create the ChatOpenAI instance with the configuration
    return ChatOpenAI(
        openai_api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_base="https://openrouter.ai/api/v1",
        extra_headers=default_headers,
        **kwargs
    )
