"""
Code template definitions and management.
"""

from typing import List, Dict, Any


class CodeTemplate:
    """Base class for code templates."""
    
    def __init__(self, name: str, description: str, template: str, variables: List[str]):
        self.name = name
        self.description = description
        self.template = template
        self.variables = variables
    
    def generate(self, **kwargs) -> str:
        """
        Generate code from template with provided variables.
        
        Args:
            **kwargs: Template variables
            
        Returns:
            Generated code as string
        """
        # Simple string formatting - can be enhanced with jinja2 if needed
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing = e.args[0]
            raise ValueError(f"Missing required template variable: {missing}") from e


# Predefined templates
TEMPLATES = {
    'flask_rest': CodeTemplate(
        name='flask_rest',
        description='Basic Flask REST API endpoint',
        template="""from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/{endpoint}', methods=['{method}'])
def {function_name}():
    # Your code here
    return jsonify({{'message': 'Hello, World!'}})

if __name__ == '__main__':
    app.run(debug=True)""",
        variables=['endpoint', 'method', 'function_name']
    ),
    'fastapi_route': CodeTemplate(
        name='fastapi_route',
        description='FastAPI route handler',
        template="""from fastapi import FastAPI

app = FastAPI()

@app.{method}('/{endpoint}')
async def {function_name}():
    # Your code here
    return {{"message": "Hello, World!"}}""",
        variables=['method', 'endpoint', 'function_name']
    )
    # Add more templates as needed
}


def get_template(name: str) -> CodeTemplate:
    """
    Get a template by name.
    
    Args:
        name: Template name
        
    Returns:
        CodeTemplate instance
        
    Raises:
        ValueError: If template not found
    """
    if name not in TEMPLATES:
        raise ValueError(f"Template '{name}' not found")
    return TEMPLATES[name]


def list_templates() -> Dict[str, str]:
    """
    List all available templates with their descriptions.
    
    Returns:
        Dict mapping template names to descriptions
    """
    return {name: template.description for name, template in TEMPLATES.items()}
