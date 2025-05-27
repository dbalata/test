"""
Code generation module for AI-assisted code generation.
"""

from .core import CodeGenerator, CodeGenerationError
from .models.generation import CodeBlock, GenerationResult, TemplateConfig
from .parser import CodeParser, ParserError, parse_code
from .templates import Template, TemplateRegistry, register_template, default_registry

# Re-export public API
__all__ = [
    # Main classes
    'CodeGenerator',
    'Template',
    'TemplateRegistry',
    'CodeParser',
    
    # Models
    'CodeBlock',
    'GenerationResult',
    'TemplateConfig',
    
    # Functions
    'parse_code',
    'register_template',
    'generate_code',
    
    # Constants
    'default_registry',
    
    # Exceptions
    'CodeGenerationError',
    'ParserError',
]

# Initialize default registry with built-in templates
_default_generator = CodeGenerator()

def generate_code(description: str, **kwargs):
    """
    Generate code using the default code generator.
    
    This is a convenience function that uses the default CodeGenerator instance.
    For more control, create and configure your own CodeGenerator instance.
    
    Args:
        description: Natural language description of the desired code
        **kwargs: Additional arguments to pass to generate_with_ai
        
    Returns:
        GenerationResult containing the generated code and metadata
    """
    return _default_generator.generate_with_ai(description, **kwargs)
