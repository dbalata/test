"""
Template registration and management.
"""

from typing import Dict, Optional, Type, List
from .base import Template
from ..models.generation import TemplateConfig


class TemplateRegistry:
    """Registry for managing code templates."""
    
    def __init__(self):
        """Initialize an empty template registry."""
        self._templates: Dict[str, Template] = {}
    
    def register(self, name: str, template: Template) -> None:
        """
        Register a new template.
        
        Args:
            name: Unique name for the template
            template: Template instance to register
            
        Raises:
            ValueError: If a template with the same name already exists
        """
        if name in self._templates:
            raise ValueError(f"Template '{name}' is already registered")
        self._templates[name] = template
    
    def get(self, name: str) -> Optional[Template]:
        """
        Get a template by name.
        
        Args:
            name: Name of the template to retrieve
            
        Returns:
            The requested template, or None if not found
        """
        return self._templates.get(name)
    
    def list_templates(self) -> Dict[str, str]:
        """
        List all registered templates.
        
        Returns:
            Dictionary mapping template names to their descriptions
        """
        return {
            name: template.config.description 
            for name, template in self._templates.items()
        }
    
    def create_template(
        self,
        name: str,
        description: str,
        template: str,
        variables: List[str],
        template_class: Optional[Type[Template]] = None
    ) -> Template:
        """
        Create and register a new template.
        
        Args:
            name: Unique name for the template
            description: Description of what the template does
            template: Template string with placeholders
            variables: List of required variable names
            template_class: Optional custom template class to use
            
        Returns:
            The created template instance
            
        Raises:
            ValueError: If a template with the same name already exists
        """
        if name in self._templates:
            raise ValueError(f"Template '{name}' is already registered")
            
        config = TemplateConfig(
            name=name,
            description=description,
            variables=variables,
            template=template
        )
        
        template_class = template_class or Template
        template_instance = template_class(config)
        self._templates[name] = template_instance
        return template_instance


def register_template(registry: Optional[TemplateRegistry] = None) -> callable:
    """
    Decorator to register a template class or function.
    
    Args:
        registry: Optional template registry to use (defaults to global registry)
    
    Returns:
        Decorator function
    """
    registry = registry or default_registry
    
    def decorator(template_or_func):
        if isinstance(template_or_func, type) and issubclass(template_or_func, Template):
            # Handle class-based template
            template = template_or_func()
            registry.register(template.config.name, template)
            return template_or_func
        else:
            # Handle function-based template
            raise NotImplementedError("Function-based templates not yet implemented")
    
    return decorator


# Global default registry
default_registry = TemplateRegistry()
