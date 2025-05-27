"""
Utility functions for the code generator.
"""

from typing import Dict, Any, List, Optional
import re

def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from text with optional language specification.
    
    Args:
        text: Input text containing code blocks
        
    Returns:
        List of dictionaries with 'language' and 'code' keys
    """
    pattern = r'```(\w*)\n([\s\S]*?)```'
    matches = re.findall(pattern, text)
    return [
        {'language': lang or 'text', 'code': code.strip()}
        for lang, code in matches
    ]

def format_dependencies(dependencies: List[str]) -> str:
    """
    Format a list of dependencies for display.
    
    Args:
        dependencies: List of dependency strings
        
    Returns:
        Formatted string of dependencies
    """
    if not dependencies:
        return ""
    return "\n".join(f"- {dep}" for dep in dependencies)

def validate_template_variables(template: str, variables: List[str]) -> bool:
    """
    Validate that all required variables are present in the template.
    
    Args:
        template: Template string
        variables: List of required variable names
        
    Returns:
        True if all variables are present, False otherwise
    """
    if not variables:
        return True
        
    for var in variables:
        pattern = f"{{{var}}}"  # Matches {var} in the template
        if pattern not in template:
            return False
    return True

def sanitize_code(code: str) -> str:
    """
    Sanitize code by removing any potentially harmful patterns.
    
    Args:
        code: Input code string
        
    Returns:
        Sanitized code string
    """
    # Remove any line that contains potentially dangerous patterns
    dangerous_patterns = [
        r'import\s+os\s*\n',
        r'import\s+subprocess\s*\n',
        r'os\.system\s*\(',
        r'subprocess\.run\s*\('
    ]
    
    for pattern in dangerous_patterns:
        code = re.sub(pattern, '', code, flags=re.IGNORECASE)
    
    return code.strip()
