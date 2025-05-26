"""
Utilities for working with OpenRouter API.
"""

import os
from typing import Dict, Any, Optional
from langchain.chat_models import ChatOpenAI


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
    return ChatOpenAI(
        model_name=model_name,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
        headers={
            "HTTP-Referer": "http://localhost:8501",  # Your site URL
            "X-Title": "LangChain App",  # Your app name
        },
        **kwargs
    )
