from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import streamlit as st

class BaseComponent(ABC):
    """Base class for all UI components.
    
    This class provides common functionality and enforces the interface
    that all UI components must implement.
    """
    
    def __init__(self):
        """Initialize the base component."""
        self._initialized = False
    
    def render(self) -> None:
        """Render the component.
        
        This method should be overridden by subclasses to implement
        the actual rendering logic.
        """
        if not self._initialized:
            self._initialize()
            self._initialized = True
        self._render()
    
    @abstractmethod
    def _render(self) -> None:
        """Internal render method to be implemented by subclasses."""
        pass
    
    def _initialize(self) -> None:
        """Initialize the component.
        
        This method is called once before the first render. Subclasses
        can override this method to perform any necessary initialization.
        """
        pass
    
    def _display_error(self, error: Exception, context: str = "") -> None:
        """Display an error message.
        
        Args:
            error: The exception that was raised.
            context: Additional context about where the error occurred.
        """
        error_msg = f"Error: {str(error)}"
        if context:
            error_msg = f"{context}: {error_msg}"
        st.error(error_msg)
