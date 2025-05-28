"""
Core code generation functionality.
"""

from typing import Dict, List, Any, Optional, Union, Type, TypeVar, Callable
from dataclasses import asdict

from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from .models.generation import GenerationResult, CodeBlock, TemplateConfig
from .parser import CodeParser, ParserError
from .templates import Template, TemplateRegistry, register_template, default_registry
from ..openrouter_utils import get_chat_openai
from .models.generation import GenerationResult

# Type alias for language model
LanguageModel = Callable[[str], str]


class CodeGenerationError(Exception):
    """Base exception for code generation errors."""
    pass


class CodeGenerator:
    """
    Main code generation system with templates and AI assistance.
    """
    
    def __init__(
        self,
        llm: Optional[LanguageModel] = None,
        template_registry: Optional[TemplateRegistry] = None,
        parser: Optional[CodeParser] = None
    ):
        """
        Initialize the code generator.
        
        Args:
            llm: Language model callable that takes a prompt and returns a string response
            template_registry: Registry for code templates
            parser: Parser for processing LLM responses
        """
        self.llm = llm or self._get_default_llm()
        self.parser = parser or CodeParser()
        self.templates = template_registry or default_registry
        self._setup_default_templates()
    
    def _get_default_llm(self) -> LanguageModel:
        """Get the default language model."""
        # This is a simple wrapper that adapts OpenRouterClient to the Callable protocol
        client = get_chat_openai()
        return lambda prompt: client.chat_complete([{"role": "user", "content": prompt}])
    
    def _setup_default_templates(self) -> None:
        """Register default templates if not already registered."""
        if not self.templates.get('python_class'):
            self.templates.create_template(
                name='python_class',
                description='Basic Python class with constructor and methods',
                template='''class {class_name}:
    """{description}"""
    
    def __init__(self{init_params}):
        {init_body}
    
    {methods}
''',
                variables=['class_name', 'description', 'init_params', 'init_body', 'methods']
            )
    
    def generate_with_ai(
        self,
        description: str,
        language: str = "python",
        framework: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate code using AI based on natural language description.
        
        Args:
            description: Natural language description of the desired code
            language: Target programming language
            framework: Optional framework (e.g., 'flask', 'django')
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            GenerationResult containing generated code and metadata
            
        Raises:
            CodeGenerationError: If code generation fails
        """
        try:
            prompt = self._build_code_generation_prompt(description, language, framework)
            response = self.llm(prompt)
            return self.parser.parse(response)
        except Exception as e:
            raise CodeGenerationError(f"Failed to generate code: {str(e)}")
    
    def _build_code_generation_prompt(
        self,
        description: str,
        language: str,
        framework: Optional[str]
    ) -> str:
        """Build the prompt for code generation."""
        framework_text = f" using {framework}" if framework else ""
        return (
            f"Generate {language} code{framework_text} for: {description}\n\n"
            "Provide:\n"
            "1. A clear explanation of the solution\n"
            "2. Well-commented code in markdown code blocks\n"
            "3. Any necessary dependencies\n"
            "4. Usage examples if applicable"
        )
    
    def explain_code(
        self,
        code: str,
        detail_level: str = "detailed",
        **kwargs
    ) -> GenerationResult:
        """
        Explain existing code with different levels of detail.
        
        Args:
            code: Source code to explain
            detail_level: Level of detail ('brief', 'detailed', 'comprehensive')
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            GenerationResult containing explanation and metadata
            
        Raises:
            CodeGenerationError: If explanation fails
        """
        try:
            prompt = self._build_code_explanation_prompt(code, detail_level)
            response = self.llm(prompt)
            return self.parser.parse(response)
        except Exception as e:
            raise CodeGenerationError(f"Failed to explain code: {str(e)}")
    
    def _build_code_explanation_prompt(
        self,
        code: str,
        detail_level: str
    ) -> str:
        """Build the prompt for code explanation."""
        return (
            f"Explain the following code with {detail_level} detail:\n\n"
            f"```\n{code}\n```\n\n"
            "Include:\n"
            "1. Purpose and functionality\n"
            "2. Key components and their roles\n"
            "3. Any important algorithms or patterns used"
        )
    
    def get_available_templates(self) -> Dict[str, str]:
        """
        Get list of available templates with descriptions.
        
        Returns:
            Dict mapping template names to descriptions
        """
        return self.templates.list_templates()
    
    def generate_from_template(
        self,
        template_name: str,
        **kwargs
    ) -> str:
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
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        return template.generate(**kwargs)
    
    # The following methods are stubs that would be implemented in separate modules
    
    def generate_api_client(
        self,
        api_description: str,
        base_url: str,
        endpoints: List[Dict[str, Any]],
        **kwargs
    ) -> GenerationResult:
        """
        Generate an API client based on specification.
        
        Args:
            api_description: Description of the API
            base_url: Base URL for the API
            endpoints: List of endpoint specifications
            **kwargs: Additional arguments for API client generation
            
        Returns:
            GenerationResult containing generated client code and metadata
        
        Raises:
            NotImplementedError: This is a stub for future implementation
        """
        raise NotImplementedError("API client generation not yet implemented")
    
    def generate_database_schema(
        self,
        requirements: str,
        database_type: str = "postgresql",
        **kwargs
    ) -> GenerationResult:
        """
        Generate database schema based on requirements.
        
        Args:
            requirements: Natural language description of database requirements
            database_type: Type of database (e.g., 'postgresql', 'mysql')
            **kwargs: Additional arguments for schema generation
            
        Returns:
            GenerationResult containing generated schema and metadata
            
        Raises:
            NotImplementedError: This is a stub for future implementation
        """
        raise NotImplementedError("Database schema generation not yet implemented")
    
    def generate_testing_suite(
        self,
        code_to_test: str,
        testing_framework: str = "pytest",
        **kwargs
    ) -> GenerationResult:
        """
        Generate test cases for the given code.
        
        Args:
            code_to_test: Source code to generate tests for
            testing_framework: Testing framework to use
            **kwargs: Additional arguments for test generation
            
        Returns:
            GenerationResult containing generated tests and metadata
            
        Raises:
            CodeGenerationError: If test generation fails
        """
        try:
            prompt = self._build_testing_suite_prompt(code_to_test, testing_framework)
            response = self.llm(prompt)
            return self.parser.parse(response)
        except Exception as e:
            raise CodeGenerationError(f"Failed to generate testing suite: {str(e)}")

    def _build_testing_suite_prompt(
        self,
        code_to_test: str,
        testing_framework: str
    ) -> str:
        """Build the prompt for test generation."""
        return (
            f"Generate a test suite using {testing_framework} for the following code:\n\n"
            f"```\n{code_to_test}\n```\n\n"
            "Provide:\n"
            "1. A brief explanation of the generated tests.\n"
            "2. The test code in markdown code blocks.\n"
            "3. Any necessary dependencies for running the tests."
        )

    def refactor_code(
        self,
        original_code: str,
        refactoring_goals: str,
        **kwargs
    ) -> GenerationResult:
        """
        Refactor existing code based on specified goals.
        
        Args:
            original_code: Source code to refactor
            refactoring_goals: Description of desired improvements
            **kwargs: Additional arguments for refactoring
            
        Returns:
            GenerationResult containing refactored code and explanation
            
        Raises:
            NotImplementedError: This is a stub for future implementation
        """
        raise NotImplementedError("Code refactoring not yet implemented")
    
    def generate_documentation(
        self,
        code: str,
        doc_type: str = "api",
        **kwargs
    ) -> GenerationResult:
        """
        Generate documentation for given code.
        
        Args:
            code: Source code to document
            doc_type: Type of documentation ('api', 'inline', 'readme')
            **kwargs: Additional arguments for documentation generation
            
        Returns:
            GenerationResult containing generated documentation and metadata
            
        Raises:
            NotImplementedError: This is a stub for future implementation
        """
        raise NotImplementedError("Documentation generation not yet implemented")
