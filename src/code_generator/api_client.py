"""
API client generation functionality.
"""

from typing import Dict, List, Any
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


def generate_api_client(
    llm,
    api_description: str,
    base_url: str,
    endpoints: List[Dict],
    language: str = "python",
    framework: str = "requests"
) -> Dict[str, Any]:
    """
    Generate an API client based on specification.
    
    Args:
        llm: Language model instance
        api_description: Description of the API
        base_url: Base URL for the API
        endpoints: List of endpoint specifications
        language: Target programming language
        framework: HTTP client framework to use
        
    Returns:
        Dict containing generated client code and metadata
    """
    prompt = PromptTemplate(
        input_variables=["api_description", "base_url", "endpoints", "language", "framework"],
        template="""Generate a {language} API client for the following API using {framework}.
        
        API Description: {api_description}
        Base URL: {base_url}
        
        Endpoints:
        {endpoints}
        
        Create a comprehensive API client class that includes:
        1. Methods for each endpoint
        2. Error handling
        3. Request/response validation
        4. Authentication if needed
        5. Proper type hints and docstrings
        
        The client should be production-ready with proper logging and error handling.
        
        Format your response with markdown code blocks."""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Format endpoints for the prompt
    formatted_endpoints = "\n".join(
        f"- {ep.get('method', 'GET')} {ep.get('path', '')}: {ep.get('description', '')}"
        for ep in endpoints
    )
    
    result = chain.run(
        api_description=api_description,
        base_url=base_url,
        endpoints=formatted_endpoints,
        language=language,
        framework=framework
    )
    
    return result
