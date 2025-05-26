"""
Code generation module with various templates and examples.
Provides functionality to generate code in multiple programming languages
with different patterns and frameworks.
"""

import os
from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from .openrouter_utils import get_openrouter_llm
import json
import re


class CodeOutputParser(BaseOutputParser):
    """Custom parser for extracting code from LLM responses."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the output to extract code, explanation, and metadata."""
        # Extract code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', text, re.DOTALL)
        
        # Extract main explanation (text before first code block)
        explanation_match = re.search(r'^(.*?)```', text, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else text.strip()
        
        # Parse different sections
        sections = {
            'explanation': explanation,
            'code_blocks': [],
            'dependencies': [],
            'usage_examples': []
        }
        
        for language, code in code_blocks:
            sections['code_blocks'].append({
                'language': language or 'text',
                'code': code.strip()
            })
        
        # Extract dependencies if mentioned
        deps_match = re.search(r'(?:dependencies|requirements|install):(.*?)(?:\n\n|\n```|$)', text, re.IGNORECASE | re.DOTALL)
        if deps_match:
            deps_text = deps_match.group(1).strip()
            sections['dependencies'] = [dep.strip() for dep in deps_text.split('\n') if dep.strip()]
        
        return sections


class CodeTemplate:
    """Base class for code templates."""
    
    def __init__(self, name: str, description: str, template: str, variables: List[str]):
        self.name = name
        self.description = description
        self.template = template
        self.variables = variables
    
    def generate(self, **kwargs) -> str:
        """Generate code from template with provided variables."""
        return self.template.format(**kwargs)


class CodeGenerator:
    """Main code generation system with templates and AI assistance."""
    
    def __init__(self):
        self.llm = get_openrouter_llm(
            model_name="openai/gpt-3.5-turbo",
            temperature=0.1
        )
        self.parser = CodeOutputParser()
        self.templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, CodeTemplate]:
        """Initialize predefined code templates."""
        templates = {}
        
        # Python class template
        templates['python_class'] = CodeTemplate(
            name="Python Class",
            description="Basic Python class with constructor and methods",
            template="""class {class_name}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self{init_params}):
        {init_body}
    
    {methods}
""",
            variables=['class_name', 'description', 'init_params', 'init_body', 'methods']
        )
        
        # REST API endpoint template
        templates['flask_api'] = CodeTemplate(
            name="Flask API Endpoint",
            description="Flask REST API endpoint with error handling",
            template="""from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({{'error': str(e)}}), 500
    return decorated_function

@app.route('/{endpoint}', methods=['{method}'])
@handle_errors
def {function_name}():
    \"\"\"
    {description}
    \"\"\"
    {function_body}
    
    return jsonify({return_value})

if __name__ == '__main__':
    app.run(debug=True)
""",
            variables=['endpoint', 'method', 'function_name', 'description', 'function_body', 'return_value']
        )
        
        # React component template
        templates['react_component'] = CodeTemplate(
            name="React Component",
            description="Functional React component with hooks",
            template="""import React, {{ useState, useEffect }} from 'react';
{additional_imports}

const {component_name} = ({props}) => {{
    {state_declarations}
    
    {effects}
    
    {helper_functions}
    
    return (
        <div className="{css_class}">
            {jsx_content}
        </div>
    );
}};

export default {component_name};
""",
            variables=['component_name', 'props', 'state_declarations', 'effects', 'helper_functions', 'css_class', 'jsx_content', 'additional_imports']
        )
        
        # Database model template
        templates['sqlalchemy_model'] = CodeTemplate(
            name="SQLAlchemy Model",
            description="SQLAlchemy database model with relationships",
            template="""from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, {additional_types}
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class {model_name}(Base):
    \"\"\"
    {description}
    \"\"\"
    __tablename__ = '{table_name}'
    
    id = Column(Integer, primary_key=True)
    {columns}
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    {relationships}
    
    def to_dict(self):
        return {{
            {to_dict_body}
        }}
    
    def __repr__(self):
        return f"<{model_name}({repr_fields})>"
""",
            variables=['model_name', 'description', 'table_name', 'columns', 'relationships', 'additional_types', 'to_dict_body', 'repr_fields']
        )
        
        # Test case template
        templates['pytest_test'] = CodeTemplate(
            name="Pytest Test Case",
            description="Comprehensive test case with fixtures and mocking",
            template="""import pytest
from unittest.mock import Mock, patch
{additional_imports}

class Test{class_name}:
    \"\"\"
    Test cases for {class_name} functionality.
    \"\"\"
    
    @pytest.fixture
    def {fixture_name}(self):
        \"\"\"Setup test fixture.\"\"\"
        {fixture_body}
        return {fixture_return}
    
    {test_methods}
    
    def test_{test_name}_error_handling(self, {fixture_name}):
        \"\"\"Test error handling scenarios.\"\"\"
        {error_test_body}
""",
            variables=['class_name', 'fixture_name', 'fixture_body', 'fixture_return', 'test_methods', 'test_name', 'error_test_body', 'additional_imports']
        )
        
        return templates
    
    def generate_from_template(self, template_name: str, **kwargs) -> Dict[str, Any]:
        """Generate code using a predefined template."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        
        template = self.templates[template_name]
        
        try:
            code = template.generate(**kwargs)
            return {
                'success': True,
                'code': code,
                'template_name': template_name,
                'description': template.description
            }
        except KeyError as e:
            return {
                'success': False,
                'error': f"Missing required variable: {e}",
                'required_variables': template.variables
            }
    
    def generate_with_ai(self, description: str, language: str = "python", framework: str = None) -> Dict[str, Any]:
        """Generate code using AI based on natural language description."""
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["description", "language", "framework"],
            template="""
Generate {language} code based on the following description:

Description: {description}
Programming Language: {language}
Framework/Library: {framework}

Please provide:
1. Complete, working code with proper error handling
2. Clear comments explaining the logic
3. Any required dependencies or imports
4. Usage examples if applicable
5. Best practices and security considerations

Format your response with:
- Brief explanation of the approach
- Main code in a code block
- Dependencies (if any)
- Usage examples

Code:
"""
        )
        
        # Create chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template, output_parser=self.parser)
        
        try:
            result = chain.run(
                description=description,
                language=language,
                framework=framework or "standard library"
            )
            
            return {
                'success': True,
                'result': result,
                'description': description,
                'language': language,
                'framework': framework
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_api_client(self, api_description: str, base_url: str, endpoints: List[Dict]) -> Dict[str, Any]:
        """Generate an API client based on specification."""
        
        prompt = f"""
Generate a Python API client for the following API:

Description: {api_description}
Base URL: {base_url}

Endpoints:
{json.dumps(endpoints, indent=2)}

Create a comprehensive API client class that includes:
1. Proper authentication handling
2. Error handling and retry logic
3. Rate limiting consideration
4. Request/response validation
5. Async support
6. Comprehensive documentation

The client should be production-ready with proper logging and error handling.
"""
        
        try:
            result = self.llm.predict(prompt)
            parsed = self.parser.parse(result)
            
            return {
                'success': True,
                'client_code': parsed,
                'api_description': api_description,
                'base_url': base_url,
                'endpoints': endpoints
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_database_schema(self, requirements: str, database_type: str = "postgresql") -> Dict[str, Any]:
        """Generate database schema based on requirements."""
        
        prompt = f"""
Design a database schema for the following requirements:

Requirements: {requirements}
Database Type: {database_type}

Provide:
1. Complete SQL CREATE statements for all tables
2. Proper indexes for performance
3. Foreign key constraints and relationships
4. Sample data insertion statements
5. Common query examples
6. Migration scripts (up/down)

Focus on:
- Normalization and data integrity
- Performance optimization
- Scalability considerations
- Security best practices
"""
        
        try:
            result = self.llm.predict(prompt)
            parsed = self.parser.parse(result)
            
            return {
                'success': True,
                'schema': parsed,
                'requirements': requirements,
                'database_type': database_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_testing_suite(self, code_to_test: str, testing_framework: str = "pytest") -> Dict[str, Any]:
        """Generate comprehensive test suite for given code."""
        
        prompt = f"""
Generate a comprehensive test suite for the following code using {testing_framework}:

Code to test:
{code_to_test}

Create tests that include:
1. Unit tests for all functions/methods
2. Integration tests for complete workflows
3. Edge cases and error conditions
4. Performance tests if applicable
5. Mock external dependencies
6. Test fixtures and setup/teardown
7. Parameterized tests for different scenarios

Follow best practices for test organization and coverage.
"""
        
        try:
            result = self.llm.predict(prompt)
            parsed = self.parser.parse(result)
            
            return {
                'success': True,
                'tests': parsed,
                'testing_framework': testing_framework,
                'original_code': code_to_test
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def refactor_code(self, original_code: str, refactoring_goals: str) -> Dict[str, Any]:
        """Refactor existing code based on specified goals."""
        
        prompt = f"""
Refactor the following code based on these goals:

Refactoring Goals: {refactoring_goals}

Original Code:
{original_code}

Provide:
1. Refactored code with improvements
2. Explanation of changes made
3. Before/after comparison highlights
4. Performance impact analysis
5. Backward compatibility considerations
6. Migration guide if breaking changes

Focus on:
- Code readability and maintainability
- Performance optimization
- Security improvements
- Best practices adherence
"""
        
        try:
            result = self.llm.predict(prompt)
            parsed = self.parser.parse(result)
            
            return {
                'success': True,
                'refactored': parsed,
                'original_code': original_code,
                'goals': refactoring_goals
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_documentation(self, code: str, doc_type: str = "api") -> Dict[str, Any]:
        """Generate documentation for given code."""
        
        prompt = f"""
Generate comprehensive {doc_type} documentation for the following code:

Code:
{code}

Create documentation that includes:
1. Overview and purpose
2. Installation instructions
3. API reference with examples
4. Usage patterns and best practices
5. Configuration options
6. Error handling and troubleshooting
7. Contributing guidelines
8. Changelog and versioning

Format as professional README.md or API documentation.
"""
        
        try:
            result = self.llm.predict(prompt)
            
            return {
                'success': True,
                'documentation': result,
                'doc_type': doc_type,
                'code': code
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_available_templates(self) -> Dict[str, str]:
        """Get list of available templates with descriptions."""
        return {name: template.description for name, template in self.templates.items()}
    
    def explain_code(self, code: str, detail_level: str = "detailed") -> Dict[str, Any]:
        """Explain existing code with different levels of detail."""
        
        detail_prompts = {
            "brief": "Provide a brief summary of what this code does",
            "detailed": "Provide a detailed explanation including logic flow, design patterns, and key concepts",
            "expert": "Provide an expert-level analysis including architecture, performance, security, and potential improvements"
        }
        
        prompt = f"""
{detail_prompts.get(detail_level, detail_prompts['detailed'])}:

Code:
{code}

Include:
1. Purpose and functionality
2. Key components and their roles
3. Logic flow and algorithms used
4. Dependencies and external integrations
5. Potential issues or improvements
6. Use cases and applications
"""
        
        try:
            result = self.llm.predict(prompt)
            
            return {
                'success': True,
                'explanation': result,
                'code': code,
                'detail_level': detail_level
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Example usage functions for demonstration
def example_flask_api_generation():
    """Example of generating a Flask API."""
    generator = CodeGenerator()
    
    result = generator.generate_from_template(
        'flask_api',
        endpoint='users',
        method='GET',
        function_name='get_users',
        description='Retrieve all users with pagination',
        function_body="""    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    users = User.query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return {
        'users': [user.to_dict() for user in users.items],
        'pagination': {
            'page': page,
            'pages': users.pages,
            'total': users.total
        }
    }""",
        return_value="{'users': users_data, 'pagination': pagination_info}"
    )
    
    return result


def example_ai_code_generation():
    """Example of AI-powered code generation."""
    generator = CodeGenerator()
    
    result = generator.generate_with_ai(
        description="Create a rate-limited API client that handles authentication, retries failed requests, and logs all API calls",
        language="python",
        framework="requests"
    )
    
    return result


if __name__ == "__main__":
    # Demo the code generation capabilities
    generator = CodeGenerator()
    
    print("Available Templates:")
    for name, desc in generator.get_available_templates().items():
        print(f"- {name}: {desc}")
    
    print("\n" + "="*50)
    print("Example: Flask API Generation")
    print("="*50)
    
    flask_result = example_flask_api_generation()
    if flask_result['success']:
        print(flask_result['code'])
    
    print("\n" + "="*50)
    print("Example: AI Code Generation")
    print("="*50)
    
    ai_result = example_ai_code_generation()
    if ai_result['success']:
        print("Generated explanation:")
        print(ai_result['result']['explanation'])
        print("\nGenerated code blocks:")
        for block in ai_result['result']['code_blocks']:
            print(f"Language: {block['language']}")
            print(block['code'][:200] + "..." if len(block['code']) > 200 else block['code'])
