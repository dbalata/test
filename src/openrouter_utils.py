"""
Utilities for working with OpenRouter API using the OpenAI client.
"""

from typing import Optional, Dict, Any, List, Union
from openai import OpenAI
import os
from dataclasses import dataclass

# Import settings
from src.config import settings

@dataclass
class OpenAIClientConfig:
    """Configuration for the OpenAI client with OpenRouter."""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "openai/gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    http_referer: str = "http://localhost:8501"
    x_title: str = "LangChain QA App"

    def get_extra_headers(self) -> Dict[str, str]:
        """Get extra headers for OpenRouter."""
        return {
            "HTTP-Referer": self.http_referer,
            "X-Title": self.x_title
        }

class OpenRouterClient:
    """A client for interacting with OpenRouter using the OpenAI client."""
    
    def __init__(self, config: Optional[OpenAIClientConfig] = None):
        """Initialize the OpenRouter client.
        
        Args:
            config: Configuration for the client. If None, uses settings from config.
        """
        if config is None:
            llm_settings = settings.llm
            self.config = OpenAIClientConfig(
                api_key=llm_settings.api_key or "dummy_key",
                base_url=str(llm_settings.base_url) if hasattr(llm_settings, 'base_url') and llm_settings.base_url else "https://openrouter.ai/api/v1",
                default_model=getattr(llm_settings, 'model', 'openai/gpt-3.5-turbo'),
                temperature=getattr(llm_settings, 'temperature', 0.7),
                max_tokens=getattr(llm_settings, 'max_tokens', None)
            )
        else:
            self.config = config
        
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key
        )
    
    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Any:
        """Generate a chat completion using OpenRouter.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: The model to use. If None, uses the default from config.
            temperature: Sampling temperature. If None, uses the default from config.
            max_tokens: Maximum number of tokens to generate. If None, uses the default from config.
            stream: Whether to stream the response. If None, uses the default from config.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            The chat completion response.
        """
        extra_headers = self.config.get_extra_headers()
        
        return self.client.chat.completions.create(
            model=model or self.config.default_model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=stream if stream is not None else self.config.stream,
            extra_headers=extra_headers,
            **kwargs
        )

def get_chat_openai(**kwargs) -> OpenRouterClient:
    """
    Get a configured OpenRouterClient instance.
    
    Args:
        **kwargs: Additional arguments to pass to the client config
        
    Returns:
        Configured OpenRouterClient instance
    """
    # Get settings
    llm_settings = settings.llm
    
    # Create config with defaults from settings, overridden by any kwargs
    config = {
        'api_key': llm_settings.api_key or "dummy_key",
        'base_url': str(llm_settings.base_url) if hasattr(llm_settings, 'base_url') and llm_settings.base_url else "https://openrouter.ai/api/v1",
        'default_model': getattr(llm_settings, 'model', 'openai/gpt-3.5-turbo'),
        'temperature': getattr(llm_settings, 'temperature', 0.7),
        'max_tokens': getattr(llm_settings, 'max_tokens', None),
    }
    
    # Update with any provided kwargs
    config.update(kwargs)
    
    # Create and return the client instance
    return OpenRouterClient(OpenAIClientConfig(**config))
