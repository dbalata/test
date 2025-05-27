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
        
        # Add LangChain compatibility
        self.input_keys = ["input"]
        self.output_keys = ["output"]
    
    def __call__(self, input_data: Union[Dict[str, Any], str, Any]) -> Dict[str, Any]:
        """Make the instance callable for LangChain compatibility.
        
        Args:
            input_data: Can be any of:
                - A dictionary with an 'input' key containing the user's message
                - A string which will be used as the user's message
                - A dictionary with a nested 'input' key (LangChain format)
                - A dictionary with a 'messages' key (OpenAI format)
                - A StringPromptValue object (from LangChain)
                - Any other object that can be converted to string
                
        Returns:
            Dictionary with the generated response in the 'output' key
        """
        # Handle StringPromptValue input (from LangChain)
        if hasattr(input_data, 'to_string') and callable(getattr(input_data, 'to_string')):
            messages = [{"role": "user", "content": input_data.to_string()}]
        # Handle string input
        elif isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        # Handle dictionary with nested 'input' key (LangChain format)
        elif isinstance(input_data, dict) and "input" in input_data and isinstance(input_data["input"], dict):
            nested_input = input_data["input"]
            if "input" in nested_input:
                messages = [{"role": "user", "content": nested_input["input"]}]
            else:
                messages = [{"role": "user", "content": str(nested_input)}]
        # Handle dictionary with 'input' key
        elif isinstance(input_data, dict) and "input" in input_data:
            messages = [{"role": "user", "content": str(input_data["input"])}]
        # Handle dictionary with 'messages' key (OpenAI format)
        elif isinstance(input_data, dict) and "messages" in input_data:
            messages = input_data["messages"]
        # Handle any other dictionary by converting it to string
        elif isinstance(input_data, dict):
            messages = [{"role": "user", "content": str(input_data)}]
        # Handle any other type by converting to string
        else:
            messages = [{"role": "user", "content": str(input_data)}]
            
        # Ensure messages is a list of dicts with 'role' and 'content' keys
        if not isinstance(messages, list):
            messages = [{"role": "user", "content": str(messages)}]
            
        # Convert each message to the correct format if needed
        formatted_messages = []
        for msg in messages if isinstance(messages, list) else [messages]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                formatted_messages.append(msg)
            elif isinstance(msg, str):
                formatted_messages.append({"role": "user", "content": msg})
            else:
                formatted_messages.append({"role": "user", "content": str(msg)})
        
        # Get the model from input_data if provided, otherwise use default
        model = (
            input_data.get("model") 
            if isinstance(input_data, dict) and "model" in input_data
            else self.config.default_model
        )
        
        # Get temperature and max_tokens from input_data if provided
        temperature = (
            input_data.get("temperature")
            if isinstance(input_data, dict) and "temperature" in input_data
            else self.config.temperature
        )
        
        max_tokens = (
            input_data.get("max_tokens")
            if isinstance(input_data, dict) and "max_tokens" in input_data
            else self.config.max_tokens
        )
        
        # Call the chat completion API
        response = self.chat_complete(
            messages=formatted_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Return the response content directly as a string
        if hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        elif hasattr(response, 'text'):
            return response.text
        else:
            return str(response)
    
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

def get_chat_openai(model_name: Optional[str] = None, **kwargs):
    """
    Get a configured OpenRouterClient instance.
    
    Args:
        model_name: The model to use (maps to default_model in config)
        **kwargs: Additional arguments to pass to the client config
        
    Returns:
        Configured OpenRouterClient instance
    """
    # Get settings
    llm_settings = settings.llm
    
    # Create config with defaults from settings
    config = {
        'api_key': llm_settings.api_key or "dummy_key",
        'base_url': str(llm_settings.base_url) if hasattr(llm_settings, 'base_url') and llm_settings.base_url else "https://openrouter.ai/api/v1",
        'default_model': model_name or getattr(llm_settings, 'model', 'openai/gpt-3.5-turbo'),
        'temperature': getattr(llm_settings, 'temperature', 0.7),
        'max_tokens': getattr(llm_settings, 'max_tokens', None),
    }
    
    # Remove model_name from kwargs if it exists to avoid duplicate parameters
    kwargs.pop('model_name', None)
    
    # Update with any provided kwargs
    config.update(kwargs)
    
    # Create and return the client instance
    return OpenRouterClient(OpenAIClientConfig(**config))
