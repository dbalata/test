"""
Code template definitions and management.

This module is kept for backward compatibility. New code should import from the
`templates` package directly.
"""

from typing import List, Dict, Any
from .templates.base import Template, TemplateError
from .templates.registry import TemplateRegistry, register_template, default_registry

# For backward compatibility
__all__ = [
    'Template',
    'TemplateError',
    'TemplateRegistry',
    'register_template',
    'default_registry',
]
