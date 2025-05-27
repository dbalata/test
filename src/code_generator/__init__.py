"""
Code generation module with various templates and examples.
Provides functionality to generate code in multiple programming languages
with different patterns and frameworks.
"""

from .parser import CodeOutputParser
from .templates import CodeTemplate, get_template, list_templates
from .core import CodeGenerator
from .api_client import generate_api_client
from .database import generate_database_schema
from .testing import generate_testing_suite
from .refactoring import refactor_code
from .documentation import generate_documentation

__all__ = [
    'CodeOutputParser',
    'CodeTemplate',
    'CodeGenerator',
    'get_template',
    'list_templates',
    'generate_api_client',
    'generate_database_schema',
    'generate_testing_suite',
    'refactor_code',
    'generate_documentation'
]
