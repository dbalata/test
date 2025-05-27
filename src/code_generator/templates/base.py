"""
Base classes for code templates.
"""

from typing import Dict, Any
from ..models.generation import TemplateConfig


class TemplateError(Exception):
    """Base exception for template-related errors."""
    pass


class Template:
    """Base class for code templates."""
    
    def __init__(self, config: TemplateConfig):
        """Initialize with template configuration."""
        self.config = config
    
    def generate(self, **kwargs) -> str:
        """
        Generate code from template with provided variables.
        
        Args:
            **kwargs: Template variables
            
        Returns:
            Generated code as string
            
        Raises:
            TemplateError: If required variables are missing
        """
        self._validate_variables(kwargs)
        try:
            return self.config.template.format(**kwargs)
        except KeyError as e:
            raise TemplateError(f"Missing template variable: {e}")
    
    def _validate_variables(self, variables: Dict[str, Any]) -> None:
        """
        Validate that all required variables are provided.
        
        Args:
            variables: Provided template variables
            
        Raises:
            TemplateError: If any required variables are missing
        """
        if not variables and self.config.variables:
            raise TemplateError(
                f"Template requires variables: {', '.join(self.config.variables)}"
            )
            
        missing = [
            var for var in self.config.variables 
            if var not in variables or variables[var] is None
        ]
        
        if missing:
            raise TemplateError(
                f"Missing required variables: {', '.join(missing)}"
            )
