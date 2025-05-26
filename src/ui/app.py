import streamlit as st
from typing import Dict, Any, Optional, Type, List

from src.ui.base_component import BaseComponent
from src.ui.components.code_generation import CodeGenerationComponent
from src.code_generator import CodeGenerator
from src.config.settings import settings

class CodeGenerationApp:
    """Main application class for the Code Generation Assistant."""
    
    def __init__(self):
        """Initialize the application."""
        self._initialize_session_state()
        self._initialize_components()
        self._setup_page_config()
    
    def _initialize_session_state(self) -> None:
        """Initialize the Streamlit session state."""
        if 'code_generator' not in st.session_state:
            st.session_state.code_generator = CodeGenerator()
    
    def _initialize_components(self) -> None:
        """Initialize UI components."""
        self.components = {
            'code_generation': CodeGenerationComponent(st.session_state.code_generator)
        }
    
    def _setup_page_config(self) -> None:
        """Set up the page configuration."""
        st.set_page_config(
            page_title=settings.APP_NAME,
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _render_sidebar(self) -> None:
        """Render the sidebar."""
        with st.sidebar:
            st.title(f"{settings.APP_NAME} 🧠")
            st.markdown("---")
            
            # Add navigation
            st.subheader("Navigation")
            self.current_view = st.radio(
                "Go to",
                ["Code Generation", "Documentation", "Settings"],
                label_visibility="collapsed"
            )
            
            # Add some useful links
            st.markdown("---")
            st.markdown("### Help & Documentation")
            st.markdown("- [Documentation](https://example.com/docs)")
            st.markdown("- [Report Issues](https://github.com/yourusername/code-gen-assistant/issues)")
    
    def _render_main_content(self) -> None:
        """Render the main content area based on the current view."""
        if self.current_view == "Code Generation":
            self.components['code_generation'].render()
        elif self.current_view == "Documentation":
            self._render_documentation()
        elif self.current_view == "Settings":
            self._render_settings()
    
    def _render_documentation(self) -> None:
        """Render the documentation view."""
        st.title("Documentation")
        st.markdown("""
        ## Code Generation Assistant
        
        This application helps you generate code using AI. You can:
        
        1. **Generate code from a description** - Describe what you want to build
        2. **Use templates** - Generate code from predefined templates
        3. **Create API clients** - Generate API client code from endpoint descriptions
        4. **Explain code** - Get explanations for existing code
        
        ### Getting Started
        
        1. Select the "Code Generation" view from the sidebar
        2. Choose your preferred generation mode
        3. Fill in the required information
        4. Click the generate button
        
        ### Features
        
        - **AI-Powered Code Generation**: Generate code using advanced AI models
        - **Templates**: Quick start with pre-built templates
        - **API Client Generation**: Generate API clients from endpoint descriptions
        - **Code Explanation**: Get detailed explanations of existing code
        """)
    
    def _render_settings(self) -> None:
        """Render the settings view."""
        st.title("Settings")
        
        st.subheader("Model Settings")
        model = st.selectbox(
            "Default Model",
            ["gpt-4", "gpt-3.5-turbo", "claude-2"],
            index=0
        )
        
        st.subheader("Code Generation")
        max_tokens = st.slider("Max Tokens", 100, 4000, 1000)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        if st.button("Save Settings"):
            # Here you would typically save these settings
            st.success("Settings saved!")
    
    def run(self) -> None:
        """Run the application."""
        self._render_sidebar()
        self._render_main_content()

# Application entry point
def main() -> None:
    """Main entry point for the application."""
    app = CodeGenerationApp()
    app.run()

if __name__ == "__main__":
    main()
