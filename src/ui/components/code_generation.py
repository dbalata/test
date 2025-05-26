from typing import Dict, Any, Optional
import streamlit as st
from src.config.settings import settings

class CodeGenerationComponent:
    """Component for handling code generation UI and logic."""
    
    def __init__(self, code_generator):
        """Initialize the code generation component.
        
        Args:
            code_generator: An instance of CodeGenerator for handling code generation logic.
        """
        self.code_generator = code_generator
    
    def render(self) -> None:
        """Render the code generation UI components."""
        st.subheader("🧩 Code Generation")
        
        # Code generation mode selection
        gen_mode = st.selectbox(
            "Generation Mode",
            ["AI Description", "Template", "API Client", "Explain Code"],
            help="Choose how you want to generate code"
        )
        
        if gen_mode == "AI Description":
            self._render_ai_description_mode()
        elif gen_mode == "Template":
            self._render_template_mode()
        elif gen_mode == "API Client":
            self._render_api_client_mode()
        elif gen_mode == "Explain Code":
            self._render_explain_code_mode()
    
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
    
    def _handle_ai_description_generation(self, description: str, language: str, framework: str) -> None:
        """Handle the AI description generation logic.
        
        Args:
            description: The description of the code to generate.
            language: The target programming language.
            framework: The target framework (optional).
        """
        try:
            result = self.code_generator.generate_with_ai(
                description, language, framework or None
            )
            if result['success']:
                self._display_generation_result(result['result'])
            else:
                st.error(f"Error: {result['error']}")
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
    
    def _display_generation_result(self, result: Dict[str, Any]) -> None:
        """Display the code generation result.
        
        Args:
            result: The result dictionary from the code generator.
        """
        st.success("Code generated successfully!")
        
        # Display explanation
        if result.get('explanation'):
            st.write("**Explanation:**")
            st.write(result['explanation'])
        
        # Display code blocks
        for i, block in enumerate(result.get('code_blocks', [])):
            st.write(f"**Code ({block.get('language', '')}):**")
            st.code(block['code'], language=block.get('language', ''))
        
        # Display dependencies
        if result.get('dependencies'):
            st.write("**Dependencies:**")
            for dep in result['dependencies']:
                st.write(f"- {dep}")
    
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
