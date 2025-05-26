"""
Code Generation UI Component

This module provides a Streamlit-based UI for generating code using AI and templates.
It handles user input, displays results, and manages the code generation workflow.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, TypedDict, Literal
import traceback

import streamlit as st
from src.config.settings import settings

# Type aliases for better code readability
CodeBlock = Dict[str, str]
CodeGenerationResult = Dict[str, Any]
CodeLanguage = str
Framework = str

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class CodeGenerationError(Exception):
    """Custom exception for code generation related errors."""
    pass

@dataclass
class GenerationResult:
    """Structured result of a code generation operation."""
    success: bool
    code_blocks: List[CodeBlock] = field(default_factory=list)
    explanation: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    error: Optional[str] = None
    raw_result: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationResult':
        """Create a GenerationResult from a dictionary."""
        return cls(
            success=data.get('success', False),
            code_blocks=data.get('code_blocks', []),
            explanation=data.get('explanation'),
            dependencies=data.get('dependencies', []),
            error=data.get('error'),
            raw_result=data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            'success': self.success,
            'code_blocks': self.code_blocks,
            'explanation': self.explanation,
            'dependencies': self.dependencies,
            'error': self.error,
            'raw_result': self.raw_result
        }

class CodeGenerationComponent:
    """Component for handling code generation UI and logic.
    
    This class provides a user interface for generating code using various methods
    including AI descriptions, templates, and API clients. It handles user input,
    manages the generation process, and displays results in a user-friendly way.
    
    Attributes:
        code_generator: The code generator instance to use for code generation.
    """
    
    def __init__(self, code_generator: Any) -> None:
        """Initialize the code generation component.
        
        Args:
            code_generator: An instance of CodeGenerator for handling code generation logic.
            
        Raises:
            ValueError: If code_generator is None or invalid.
        """
        if code_generator is None:
            raise ValueError("code_generator cannot be None")
            
        self.code_generator = code_generator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def render(self) -> None:
        """Render the code generation UI components.
        
        This method serves as the main entry point for rendering the UI.
        It displays the mode selection and delegates to the appropriate
        rendering method based on the user's selection.
        """
        try:
            st.subheader("🧩 Code Generation")
            
            # Code generation mode selection
            gen_mode = st.selectbox(
                "Generation Mode",
                ["AI Description", "Template", "API Client", "Explain Code"],
                help="Choose how you want to generate code"
            )
            
            # Map modes to their corresponding render methods
            mode_handlers = {
                "AI Description": self._render_ai_description_mode,
                "Template": self._render_template_mode,
                "API Client": self._render_api_client_mode,
                "Explain Code": self._render_explain_code_mode
            }
            
            # Call the appropriate handler
            if gen_mode in mode_handlers:
                mode_handlers[gen_mode]()
            else:
                self.logger.warning(f"Unknown generation mode: {gen_mode}")
                st.warning(f"Generation mode '{gen_mode}' is not supported.")
                
        except Exception as e:
            error_msg = f"Error rendering code generation UI: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            st.error("An error occurred while rendering the code generation interface.")
            if settings.DEBUG:
                st.exception(e)
    
    def _render_ai_description_mode(self) -> None:
        """Render UI for AI description-based code generation."""
        description = st.text_area(
            "Describe what you want to build:",
            placeholder="Create a REST API endpoint that handles user authentication with JWT tokens"
        )
        language = st.selectbox("Language", ["python", "javascript", "sql", "other"])
        framework = st.text_input("Framework (optional)", placeholder="flask, react, fastapi")
        
        if st.button("🚀 Generate Code"):
            if description:
                with st.spinner("Generating code..."):
                    self._handle_ai_description_generation(description, language, framework)
            else:
                st.warning("Please provide a description")
    
    def _handle_ai_description_generation(self, description: str, language: str, framework: Optional[str] = None) -> None:
        """Handle the AI description generation logic.
        
        This method processes the user's description, sends it to the AI code generator,
        and handles the response. It validates inputs, processes the result, and displays
        the generated code or error messages.
        
        Args:
            description: The natural language description of the desired code.
            language: The target programming language for code generation.
            framework: Optional framework or library to use (e.g., 'flask', 'react').
            
        Raises:
            CodeGenerationError: If there's a problem with the generation process.
        """
        self.logger.info("Starting AI description-based code generation")
        self.logger.debug(
            "Generation parameters - Language: %s, Framework: %s",
            language,
            framework or 'None'
        )
        
        if not description.strip():
            error_msg = "Description cannot be empty"
            self.logger.warning(error_msg)
            st.error(error_msg)
            return
            
        if not language.strip():
            error_msg = "Programming language must be specified"
            self.logger.warning(error_msg)
            st.error(error_msg)
            return
        
        try:
            # Normalize framework to None if empty string
            framework = framework.strip() if framework else None
            
            with st.spinner("Generating code..."):
                self.logger.info("Initiating code generation with AI...")
                raw_result = self.code_generator.generate_with_ai(
                    description=description,
                    language=language,
                    framework=framework
                )
                
                self.logger.debug("Raw generation result: %s", 
                                 json.dumps(raw_result, indent=2, default=str))
                
                # Process the result
                try:
                    result = GenerationResult.from_dict(raw_result)
                    
                    if not result.success:
                        error_msg = result.error or "Unknown error occurred during code generation"
                        self.logger.error("Generation failed: %s", error_msg)
                        st.error(f"Error: {error_msg}")
                        return
                        
                    self.logger.info("Code generation completed successfully")
                    self._display_generation_result(result)
                    
                except Exception as parse_error:
                    self.logger.error(
                        "Error processing generation result: %s",
                        str(parse_error),
                        exc_info=True
                    )
                    # Fall back to displaying raw result if processing fails
                    self._display_raw_result(raw_result)
        
        except Exception as e:
            error_msg = f"Failed to generate code: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            st.error("An error occurred during code generation.")
            if settings.DEBUG:
                st.exception(e)
            
            # Provide more user-friendly error for common cases
            if "API key" in str(e):
                st.error("Authentication failed. Please check your API key configuration.")
            elif "connection" in str(e).lower():
                st.error("Unable to connect to the code generation service. Please check your internet connection.")
    
    def _display_raw_result(self, raw_result: Any) -> None:
        """Display a raw result when structured processing fails.
        
        Args:
            raw_result: The raw result to display.
        """
        self.logger.warning("Displaying raw result due to processing error")
        st.warning("Received an unexpected response format. Showing raw result:")
        
        if isinstance(raw_result, (dict, list)):
            st.json(raw_result)
        else:
            st.code(str(raw_result), language='text')
    
    def _display_generation_result(self, result: Union[Dict[str, Any], GenerationResult]) -> None:
        """Display the code generation result in a user-friendly format.
        
        This method handles different response formats and displays the generated
        code along with any explanations, dependencies, or other metadata.
        
        Args:
            result: The generation result to display, either as a dictionary or GenerationResult.
        """
        self.logger.info("Rendering generation result")
        
        # Convert to GenerationResult if it's a dict
        if isinstance(result, dict):
            try:
                result = GenerationResult.from_dict(result)
            except Exception as e:
                self.logger.error("Error converting dict to GenerationResult: %s", str(e))
                self._display_raw_result(result)
                return
        
        if not result or not result.code_blocks:
            self.logger.warning("No code blocks found in the result")
            st.warning("No code was generated. Please try again with a different description.")
            return
            
        try:
            # Display success message
            st.success("✅ Code generated successfully!")
            
            # Display explanation if available
            if result.explanation:
                with st.expander("📝 Explanation", expanded=True):
                    st.markdown(result.explanation)
            
            # Display code blocks with tabs for multiple files
            if len(result.code_blocks) > 1:
                tabs = st.tabs([f"File {i+1}" for i in range(len(result.code_blocks))])
                for i, (tab, block) in enumerate(zip(tabs, result.code_blocks), 1):
                    with tab:
                        self._display_code_block(block, f"Generated Code {i}")
            else:
                self._display_code_block(result.code_blocks[0], "Generated Code")
            
            # Display dependencies if available
            if result.dependencies:
                with st.expander("📦 Dependencies", expanded=False):
                    st.write("The following dependencies are required:")
                    for dep in result.dependencies:
                        st.code(dep, language='bash' if 'pip install' in str(dep) else 'text')
            
            # Add copy to clipboard functionality
            if len(result.code_blocks) == 1 and 'code' in result.code_blocks[0]:
                code = result.code_blocks[0]['code']
                st.download_button(
                    label="📋 Download Code",
                    data=code,
                    file_name=f"generated_code.{result.code_blocks[0].get('language', 'txt')}",
                    mime="text/plain"
                )
                
        except Exception as e:
            error_msg = f"Error displaying result: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            st.error("An error occurred while displaying the result.")
            if settings.DEBUG:
                st.exception(e)
    
    def _display_code_block(self, block: Dict[str, str], title: str = "Code") -> None:
        """Display a single code block with proper formatting.
        
        Args:
            block: Dictionary containing 'code' and optionally 'language' and 'filename'
            title: Title to display above the code block
        """
        if not block or 'code' not in block:
            self.logger.warning("Invalid code block format")
            return
            
        language = block.get('language', '').lower()
        filename = block.get('filename', '')
        code = block['code']
        
        # Display filename if available
        if filename:
            st.caption(f"File: {filename}")
        
        # Determine language for syntax highlighting
        if not language and filename:
            # Try to guess language from file extension
            ext = filename.split('.')[-1].lower()
            language = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'java': 'java',
                'c': 'c',
                'cpp': 'cpp',
                'cs': 'csharp',
                'go': 'go',
                'rb': 'ruby',
                'rs': 'rust',
                'sh': 'bash',
                'html': 'html',
                'css': 'css',
                'json': 'json',
                'yaml': 'yaml',
                'toml': 'toml',
                'md': 'markdown',
            }.get(ext, '')
        
        # Display the code with syntax highlighting
        st.code(code, language=language or None)
    
    def _render_template_mode(self) -> None:
        """Render UI for template-based code generation."""
        templates = self.code_generator.get_available_templates()
        template_name = st.selectbox("Select Template", list(templates.keys()))
        
        if template_name:
            st.write(f"**Description:** {templates[template_name]}")
            
            if template_name == "python_class":
                self._render_python_class_template()
            elif template_name == "flask_api":
                self._render_flask_api_template()
    
    def _render_python_class_template(self) -> None:
        """Render UI for Python class template."""
        class_name = st.text_input("Class Name", "MyClass")
        description = st.text_input("Description", "A sample class")
        
        if st.button("Generate from Template"):
            try:
                result = self.code_generator.generate_from_template(
                    "python_class",
                    class_name=class_name,
                    description=description,
                    init_params="",
                    init_body="pass",
                    methods="def sample_method(self):\n        pass"
                )
                if result['success']:
                    st.code(result['code'], language='python')
                else:
                    st.error(result['error'])
            except Exception as e:
                st.error(f"Template generation failed: {str(e)}")
    
    def _render_flask_api_template(self) -> None:
        """Render UI for Flask API template."""
        endpoint = st.text_input("Endpoint", "users")
        method = st.selectbox("Method", ["GET", "POST", "PUT", "DELETE"])
        function_name = st.text_input("Function Name", "handle_users")
        
        if st.button("Generate from Template"):
            try:
                result = self.code_generator.generate_from_template(
                    "flask_api",
                    endpoint=endpoint,
                    method=method,
                    function_name=function_name,
                    description=f"Handle {method} requests for {endpoint}",
                    function_body="    # Add your logic here\n    pass",
                    return_value="{'status': 'success'}"
                )
                if result['success']:
                    st.code(result['code'], language='python')
                else:
                    st.error(result['error'])
            except Exception as e:
                st.error(f"Template generation failed: {str(e)}")
    
    def _render_api_client_mode(self) -> None:
        """Render UI for API client generation."""
        api_description = st.text_input("API Description", "User Management API")
        base_url = st.text_input("Base URL", "https://api.example.com")
        endpoints_text = st.text_area(
            "Endpoints (one per line)",
            "GET /users - Get all users\nPOST /users - Create user\nGET /users/{id} - Get user by ID"
        )
        
        if st.button("Generate API Client"):
            if api_description and base_url and endpoints_text:
                with st.spinner("Generating API client..."):
                    self._handle_api_client_generation(api_description, base_url, endpoints_text)
            else:
                st.warning("Please fill in all fields")
    
    def _handle_api_client_generation(self, api_description: str, base_url: str, endpoints_text: str) -> None:
        """Handle the API client generation logic.
        
        Args:
            api_description: Description of the API.
            base_url: Base URL for the API.
            endpoints_text: Text containing API endpoints (one per line).
        """
        try:
            # Parse endpoints
            endpoints = []
            for line in endpoints_text.split('\n'):
                if line.strip() and ' ' in line:
                    parts = line.split(' ', 2)
                    if len(parts) >= 2:
                        endpoints.append({
                            'method': parts[0],
                            'path': parts[1],
                            'description': parts[2] if len(parts) > 2 else ''
                        })
            
            result = self.code_generator.generate_api_client(
                api_description, base_url, endpoints
            )
            
            if result['success']:
                st.success("API client generated!")
                self._display_generation_result(result['client_code'])
            else:
                st.error(f"Error: {result['error']}")
        except Exception as e:
            st.error(f"API client generation failed: {str(e)}")
    
    def _render_explain_code_mode(self) -> None:
        """Render UI for code explanation."""
        code_input = st.text_area(
            "Paste your code here:",
            placeholder="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        )
        detail_level = st.selectbox("Detail Level", ["brief", "detailed", "expert"])
        
        if st.button("Explain Code"):
            if code_input:
                with st.spinner("Analyzing code..."):
                    self._handle_code_explanation(code_input, detail_level)
            else:
                st.warning("Please provide some code to explain")
    
    def _handle_code_explanation(self, code_input: str, detail_level: str) -> None:
        """Handle the code explanation logic.
        
        Args:
            code_input: The code to explain.
            detail_level: The level of detail for the explanation.
        """
        try:
            result = self.code_generator.explain_code(code_input, detail_level)
            if result['success']:
                self._display_generation_result(result['result'])
            else:
                st.error(f"Error: {result['error']}")
        except Exception as e:
            st.error(f"Code explanation failed: {str(e)}")
