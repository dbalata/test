"""
Template system for code generation.
"""

from typing import Dict, Optional

from .base import Template, TemplateError
from .registry import TemplateRegistry, register_template, default_registry

# Re-export public API
__all__ = [
    'Template',
    'TemplateError',
    'TemplateRegistry',
    'register_template',
    'default_registry',
    'get_template',
    'list_templates',
]


def get_template(name: str) -> Template:
    """
    Get a template by name from the default registry.
    
    Args:
        name: Name of the template to retrieve
        
    Returns:
        The requested template
        
    Raises:
        ValueError: If the template is not found
    """
    template = default_registry.get(name)
    if template is None:
        raise ValueError(f"Template '{name}' not found")
    return template


def list_templates() -> Dict[str, str]:
    """
    List all available templates in the default registry.
    
    Returns:
        Dictionary mapping template names to their descriptions
    """
    return default_registry.list_templates()
