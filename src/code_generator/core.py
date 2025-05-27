"""
Core code generation functionality.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain.chains.llm import LLMChain

from .parser import CodeOutputParser
from .templates import get_template, list_templates
from .api_client import generate_api_client
from .database import generate_database_schema
from .testing import generate_testing_suite
from .refactoring import refactor_code
from .documentation import generate_documentation
from src.openrouter_utils import OpenRouterClient

# Type alias for the language model
LanguageModel = Union[OpenRouterClient, Callable]  # Accept either OpenRouterClient or callable


class CodeGenerator:
    """
    Main code generation system with templates and AI assistance.
    """
    
    def __init__(self, llm: Optional[LanguageModel] = None):
        """
        Initialize the code generator.
        
        Args:
            llm: Optional language model instance. If not provided, a default will be used.
        """
        self.llm = llm or get_chat_openai()
        self.parser = CodeOutputParser()
        self._setup_prompts()
    
    def _setup_prompts(self) -> None:
        """Initialize prompt templates."""
        self.code_generation_prompt = PromptTemplate(
            input_variables=["description", "language", "framework"],
            template="""Generate {language} code for the following task. {framework}
            
            Task: {description}
            
            Provide:
            1. A clear explanation of the solution
            2. Well-commented code
            3. Any necessary dependencies
            
            Format your response with markdown code blocks."""
        )
        
        self.code_explanation_prompt = PromptTemplate(
            input_variables=["code", "detail_level"],
            template="""Explain the following code with {detail_level} detail:
            
            ```
            {code}
            ```
            
            Include:
            1. Purpose and functionality
            2. Key components and their roles
            3. Any important algorithms or patterns used"""
        )
    
    def generate_with_ai(self, description: str, language: str = "python", 
                        framework: str = None) -> Dict[str, Any]:
        """
        Generate code using AI based on natural language description.
        
        Args:
            description: Natural language description of the desired code
            language: Target programming language
            framework: Optional framework (e.g., 'flask', 'django')
            
        Returns:
            Dict containing generated code and metadata
        """
        framework_text = f"Use {framework} framework." if framework else ""
        
        chain = LLMChain(
            llm=self.llm,
            prompt=self.code_generation_prompt
        )
        
        result = chain.run(
            description=description,
            language=language,
            framework=framework_text
        )
        
        return self.parser.parse(result)
    
    def explain_code(self, code: str, detail_level: str = "detailed") -> Dict[str, Any]:
        """
        Explain existing code with different levels of detail.
        
        Args:
            code: Source code to explain
            detail_level: Level of detail ('brief', 'detailed', 'comprehensive')
            
        Returns:
            Dict containing explanation and metadata
        """
        chain = LLMChain(
            llm=self.llm,
            prompt=self.code_explanation_prompt
        )
        
        result = chain.run(
            code=code,
            detail_level=detail_level
        )
        
        return self.parser.parse(result)
    
    def get_available_templates(self) -> Dict[str, str]:
        """
        Get list of available templates with descriptions.
        
        Returns:
            Dict mapping template names to descriptions
        """
        return list_templates()
    
    def generate_from_template(self, template_name: str, **kwargs) -> str:
        """
        Generate code using a predefined template.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Template variables
            
        Returns:
            Generated code as string
            
        Raises:
            ValueError: If template not found or required variables missing
        """
        template = get_template(template_name)
        return template.generate(**kwargs)
        
    def generate_api_client(self, api_description: str, base_url: str, 
                          endpoints: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Generate an API client based on specification.
        
        Args:
            api_description: Description of the API
            base_url: Base URL for the API
            endpoints: List of endpoint specifications
            **kwargs: Additional arguments for API client generation
            
        Returns:
            Dict containing generated client code and metadata
        """
        return generate_api_client(
            llm=self.llm,
            api_description=api_description,
            base_url=base_url,
            endpoints=endpoints,
            **kwargs
        )
    
    def generate_database_schema(self, requirements: str, 
                               database_type: str = "postgresql") -> Dict[str, Any]:
        """
        Generate database schema based on requirements.
        
        Args:
            requirements: Natural language description of database requirements
            database_type: Type of database (e.g., 'postgresql', 'mysql')
            
        Returns:
            Dict containing generated schema and metadata
        """
        return generate_database_schema(
            llm=self.llm,
            requirements=requirements,
            database_type=database_type
        )
    
    def generate_testing_suite(self, code_to_test: str, 
                             testing_framework: str = "pytest") -> Dict[str, Any]:
        """
        Generate test cases for the given code.
        
        Args:
            code_to_test: Source code to generate tests for
            testing_framework: Testing framework to use
            
        Returns:
            Dict containing generated tests and metadata
        """
        return generate_testing_suite(
            llm=self.llm,
            code_to_test=code_to_test,
            testing_framework=testing_framework
        )
    
    def refactor_code(self, original_code: str, 
                     refactoring_goals: str) -> Dict[str, Any]:
        """
        Refactor existing code based on specified goals.
        
        Args:
            original_code: Source code to refactor
            refactoring_goals: Description of desired improvements
            
        Returns:
            Dict containing refactored code and explanation
        """
        return refactor_code(
            llm=self.llm,
            original_code=original_code,
            refactoring_goals=refactoring_goals
        )
    
    def generate_documentation(self, code: str, 
                             doc_type: str = "api") -> Dict[str, Any]:
        """
        Generate documentation for given code.
        
        Args:
            code: Source code to document
            doc_type: Type of documentation ('api', 'inline', 'readme')
            
        Returns:
            Dict containing generated documentation and metadata
        """
        return generate_documentation(
            llm=self.llm,
            code=code,
            doc_type=doc_type
        )
