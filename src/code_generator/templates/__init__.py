"""
Template system for code generation.
"""

from .base import Template, TemplateError
from .registry import TemplateRegistry, register_template

__all__ = ['Template', 'TemplateError', 'TemplateRegistry', 'register_template']
