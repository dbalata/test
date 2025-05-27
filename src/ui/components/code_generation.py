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
from src.code_generator import CodeOutputParser

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
        self.code_parser = CodeOutputParser()
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
        handles the response, and displays the results. It validates inputs, processes 
        the result, and displays the generated code or error messages.
        
        Args:
            description: The natural language description of the desired code.
            language: The target programming language for code generation.
            framework: Optional framework or library to use (e.g., 'flask', 'react').
            
        Raises:
            CodeGenerationError: If there's a problem with the generation process.
        """
        if not description or not language:
            raise CodeGenerationError("Description and language are required")
            
        try:
            with st.spinner("Generating code..."):
                # Call the code generator with the provided parameters
                raw_result = self.code_generator.generate_with_ai(
                    description=description,
                    language=language,
                    framework=framework
                )
                
                # Parse the result using the code parser
                parsed_result = self.code_parser.parse(raw_result)
                
                # Process the parsed result
                if parsed_result.get('error'):
                    st.error(f"Error generating code: {parsed_result['error']}")
                    return
                
                # Display the parsed result
                self._display_generation_result(parsed_result)
                    
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            self.logger.exception(error_msg)
            st.error(error_msg)
            raise CodeGenerationError(error_msg) from e
            
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
        try:
            # Convert to GenerationResult if it's a dict
            if isinstance(result, dict):
                result = GenerationResult.from_dict(result)
            
            # Display success message
            st.success("✅ Code generated successfully!")
            
            # Display explanation if available
            if result.explanation:
                with st.expander("📝 Explanation", expanded=True):
                    st.markdown(result.explanation)
            
            # Display code blocks with tabs for multiple files
            if hasattr(result, 'code_blocks') and result.code_blocks:
                if len(result.code_blocks) > 1:
                    tabs = st.tabs([f"File {i+1}" for i in range(len(result.code_blocks))])
                    for i, (tab, block) in enumerate(zip(tabs, result.code_blocks), 1):
                        with tab:
                            self._display_code_block(block, f"Generated Code {i}")
                else:
                    self._display_code_block(result.code_blocks[0], "Generated Code")
            
            # Display dependencies if available
            if hasattr(result, 'dependencies') and result.dependencies:
                with st.expander("📦 Dependencies", expanded=False):
                    st.write("The following dependencies are required:")
                    if isinstance(result.dependencies, list):
                        for dep in result.dependencies:
                            st.code(dep, language='bash' if 'pip install' in str(dep) else 'text')
                    elif isinstance(result.dependencies, dict):
                        for name, version in result.dependencies.items():
                            st.code(f"{name}{'@' + str(version) if version else ''}", language='text')
            
            # Add copy to clipboard functionality
            if hasattr(result, 'code_blocks') and result.code_blocks:
                if st.button("📋 Copy to Clipboard"):
                    all_code = "\n\n".join(
                        f"# {block.get('filename', f'Code Block {i+1}')}\n{block.get('code', '')}"
                        for i, block in enumerate(result.code_blocks)
                        if isinstance(block, dict)
                    )
                    try:
                        import pyperclip
                        pyperclip.copy(all_code)
                        st.success("Code copied to clipboard!")
                    except ImportError:
                        st.warning("pyperclip not installed. Install with 'pip install pyperclip' for copy functionality.")
            
            # Add download functionality
            if hasattr(result, 'code_blocks') and result.code_blocks:
                if len(result.code_blocks) == 1 and isinstance(result.code_blocks[0], dict) and 'code' in result.code_blocks[0]:
                    # Single file download
                    code = result.code_blocks[0]['code']
                    filename = result.code_blocks[0].get('filename', 'generated_code.py')
                    st.download_button(
                        label="💾 Download Code",
                        data=code,
                        file_name=filename,
                        mime="text/plain"
                    )
                elif len(result.code_blocks) > 1:
                    # Multiple files - create a zip
                    try:
                        import io
                        import zipfile
                        
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for i, block in enumerate(block for block in result.code_blocks if isinstance(block, dict) and 'code' in block):
                                filename = block.get('filename', f'code_{i+1}.py')
                                zip_file.writestr(filename, block['code'])
                        
                        st.download_button(
                            label="💾 Download All Files",
                            data=zip_buffer.getvalue(),
                            file_name="generated_code.zip",
                            mime="application/zip"
                        )
                    except Exception as zip_error:
                        self.logger.error(f"Error creating zip file: {str(zip_error)}")
                        st.warning("Could not create zip file. Please try downloading individual files.")
            
            # Display any additional metadata
            if hasattr(result, 'metadata') and result.metadata:
                with st.expander("🔍 Additional Metadata"):
                    st.json(result.metadata)
            
            # Display raw result in debug mode
            if settings.DEBUG and hasattr(result, 'raw_result') and result.raw_result is not None:
                with st.expander("🐛 Debug: Raw Result"):
                    st.json(result.raw_result)
                    
        except Exception as e:
            error_msg = f"Error displaying generation result: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            st.error("An error occurred while displaying the result.")
            if settings.DEBUG:
                st.exception(e)
    
    def _display_code_block(self, block: Dict[str, str], title: str = "Code") -> None:
        """Display a single code block with proper formatting.
        
        Args:
            block: Dictionary containing 'code' and optionally 'language' and 'filename'
            title: Title to display above the code block
            
        Note:
            Generated files should not exceed 200 lines to maintain readability and maintainability.
            Consider breaking down larger files into smaller, focused modules.
        """
        if not block or 'code' not in block:
            self.logger.warning("Invalid code block format")
            return
            
        language = block.get('language', '').lower()
        filename = block.get('filename', '')
        code = block['code']
        
        # Check code length
        line_count = len(code.splitlines())
        if line_count > 200:
            st.warning(
                f"⚠️ This generated code is {line_count} lines long. "
                "For better maintainability, consider breaking it down into smaller, "
                "focused modules of 200 lines or less."
            )
        
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
        # Get available templates from the code generator
        try:
            templates = self.code_generator.get_available_templates()
            
            if not templates:
                st.warning("No templates available. Please check your configuration.")
                return
                
            # Create a user-friendly display name for each template
            template_options = [
                f"{name}: {info.get('description', 'No description')}" 
                for name, info in templates.items()
            ]
            
            selected_template = st.selectbox(
                "Select a template",
                options=template_options,
                format_func=lambda x: x.split(":")[0]  # Show only the template name in the dropdown
            )
            
            # Extract the template name from the selected option
            template_name = selected_template.split(":")[0].strip()
            template_info = templates.get(template_name, {})
            
            # Display template description
            st.markdown(f"**Description:** {template_info.get('description', 'No description available')}")
            
            # Get template parameters
            params = {}
            for param_name, param_info in template_info.get('parameters', {}).items():
                param_type = param_info.get('type', 'str')
                param_default = param_info.get('default', '')
                param_description = param_info.get('description', '')
                
                # Create appropriate input based on parameter type
                if param_type == 'str':
                    params[param_name] = st.text_input(
                        f"{param_name} ({param_type})",
                        value=param_default,
                        help=param_description
                    )
                elif param_type == 'int':
                    params[param_name] = st.number_input(
                        f"{param_name} ({param_type})",
                        value=int(param_default) if param_default else 0,
                        help=param_description
                    )
                elif param_type == 'bool':
                    params[param_name] = st.checkbox(
                        f"{param_name} ({param_type})",
                        value=param_default.lower() == 'true' if param_default else False,
                        help=param_description
                    )
                else:
                    # Default to text input for unknown types
                    params[param_name] = st.text_input(
                        f"{param_name} ({param_type})",
                        value=str(param_default),
                        help=param_description
                    )
            
            if st.button("Generate Code"):
                with st.spinner(f"Generating {template_name}..."):
                    try:
                        # Generate code from template
                        generated_code = self.code_generator.generate_from_template(
                            template_name,
                            **params
                        )
                        
                        # Parse the result using the code parser
                        parsed_result = self.code_parser.parse(generated_code)
                        
                        # Process the parsed result
                        if parsed_result.get('error'):
                            st.error(f"Error generating code: {parsed_result['error']}")
                        else:
                            self._display_generation_result(parsed_result)
                            
                    except Exception as e:
                        error_msg = f"Error generating code from template: {str(e)}"
                        self.logger.exception(error_msg)
                        st.error(error_msg)
                        
        except Exception as e:
            error_msg = f"Failed to load templates: {str(e)}"
            self.logger.exception(error_msg)
            st.error(error_msg)
    
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
        if not api_description or not base_url or not endpoints_text:
            st.error("API description, base URL, and at least one endpoint are required")
            return
            
        try:
            # Parse endpoints from text
            endpoints = []
            for line in endpoints_text.strip().split('\n'):
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if len(parts) >= 2:
                        method = parts[0].upper()
                        path = parts[1]
                        description = ' '.join(parts[2:]) if len(parts) > 2 else ''
                        endpoints.append({
                            'method': method,
                            'path': path,
                            'description': description
                        })
            
            if not endpoints:
                st.error("No valid endpoints found. Format: 'METHOD /path [description]'")
                return
                
            with st.spinner("Generating API client..."):
                # Generate the API client
                generated_code = self.code_generator.generate_api_client(
                    api_description=api_description,
                    base_url=base_url,
                    endpoints=endpoints
                )
                
                # Parse the result using the code parser
                parsed_result = self.code_parser.parse(generated_code)
                
                # Process the parsed result
                if parsed_result.get('error'):
                    st.error(f"Error generating API client: {parsed_result['error']}")
                else:
                    self._display_generation_result(parsed_result)
                    
        except Exception as e:
            error_msg = f"An error occurred while generating the API client: {str(e)}"
            self.logger.exception(error_msg)
            st.error(error_msg)
        
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
        if not code_input.strip():
            st.warning("Please enter some code to explain.")
            return
            
        try:
            with st.spinner("Analyzing code..."):
                # Generate the explanation
                explanation = self.code_generator.explain_code(code_input, detail_level=detail_level)
                
                # Parse the result using the code parser
                parsed_result = self.code_parser.parse(explanation)
                
                # Process the parsed result
                if parsed_result.get('error'):
                    st.error(f"Error explaining code: {parsed_result['error']}")
                else:
                    self._display_generation_result(parsed_result)
                    
        except Exception as e:
            error_msg = f"An error occurred while explaining the code: {str(e)}"
            self.logger.exception(error_msg)
            st.error(error_msg)
