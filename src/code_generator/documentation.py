"""
Documentation generation functionality.
"""

from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


def generate_documentation(
    llm,
    code: str,
    doc_type: str = "api",
    language: str = "python"
) -> Dict[str, Any]:
    """
    Generate documentation for given code.
    
    Args:
        llm: Language model instance
        code: Source code to document
        doc_type: Type of documentation ('api', 'inline', 'readme')
        language: Programming language of the code
        
    Returns:
        Dict containing generated documentation and metadata
    """
    doc_templates = {
        "api": """Generate API documentation for the following {language} code.
        Include:
        1. Module/package description
        2. Function/method signatures with parameters and return types
        3. Example usage
        4. Error handling
        
        Code:
        ```{language}
        {code}
        ```""",
        "inline": """Add clear and concise docstrings to the following {language} code.
        Follow the standard documentation style for the language.
        
        Code:
        ```{language}
        {code}
        ```""",
        "readme": """Create a comprehensive README for the following {language} code.
        Include:
        1. Project description
        2. Installation instructions
        3. Usage examples
        4. Configuration options
        5. Contributing guidelines
        
        Code:
        ```{language}
        {code}
        ```"""
    }
    
    if doc_type not in doc_templates:
        raise ValueError(f"Unsupported documentation type: {doc_type}")
    
    prompt = PromptTemplate(
        input_variables=["code", "language"],
        template=doc_templates[doc_type]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(code=code, language=language)
    return result
