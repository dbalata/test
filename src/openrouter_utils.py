"""
Utilities for working with OpenRouter API.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache, get_llm_cache

# Import settings first to avoid circular imports
from src.config.settings import settings

# Set up in-memory cache for LLM responses
cache = get_llm_cache()
if cache is None:
    cache = InMemoryCache()
    set_llm_cache(cache)

# Workaround for Pydantic validation issue with ChatOpenAI
# We'll use a simple dictionary-based configuration instead of direct instantiation


def get_openrouter_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> ChatOpenAI:
    """
    Initialize a ChatOpenAI instance configured for OpenRouter.
    
    Args:
        model_name: The model to use (e.g., 'gpt-4-turbo'). 
                   If not provided, uses the model from settings.
        temperature: The temperature to use for sampling. Uses settings if not provided.
        max_tokens: Maximum number of tokens to generate. Uses settings if not provided.
        **kwargs: Additional arguments to pass to ChatOpenAI
        
    Returns:
        Configured ChatOpenAI instance
    """
    # Use settings values as defaults
    llm_settings = settings.llm
    model_name = model_name or llm_settings.model
    temperature = temperature if temperature is not None else llm_settings.temperature
    max_tokens = max_tokens or llm_settings.max_tokens
    
    # Get API key from settings or environment
    api_key = llm_settings.api_key or os.environ.get("LANGCHAIN_QA_LLM_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenRouter API key is not configured. "
            "Please set LANGCHAIN_QA_LLM_API_KEY in your environment or .env file."
        )
    
    # Configure default headers for OpenRouter
    headers = kwargs.pop('headers', {})
    headers.update({
        "HTTP-Referer": "http://localhost:8501",  # Your site URL
        "X-Title": "LangChain App",  # Your app name
    })
    
    # Create a minimal configuration
    config = {
        "openai_api_key": api_key,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "openai_api_base": llm_settings.base_url,
        "extra_headers": headers,
        "streaming": False,  # Disable streaming to avoid issues
        **kwargs
    }
    
    # Create the ChatOpenAI instance with minimal configuration
    chat = ChatOpenAI(**config)
    
    # Ensure the disable_streaming attribute exists
    if not hasattr(chat, 'disable_streaming'):
        chat.disable_streaming = False
        
    return chat
