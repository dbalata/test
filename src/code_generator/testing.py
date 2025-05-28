"""
Testing utilities for code generation.

This module provides functions for generating test cases and test suites for generated code.
"""

from .core import CodeGenerator

def generate_testing_suite(code: str, language: str = "python") -> str:
    """
    Generate a test suite for the given code.
    
    Args:
        code: The source code to generate tests for
        language: The programming language of the code
        
    Returns:
        A string containing the generated test suite, or a comment if none could be generated.
    """
    code_generator = CodeGenerator()
    
    testing_framework = "pytest" # Default
    if language == "python":
        testing_framework = "pytest"
    elif language == "javascript":
        testing_framework = "jest"
    # Add more language-to-framework mappings as needed
    
    try:
        result = code_generator.generate_testing_suite(
            code_to_test=code,
            testing_framework=testing_framework
        )
        
        if result.code_blocks:
            # Assuming the first code block contains the test suite
            return result.code_blocks[0].code
        else:
            return "# No test suite code was generated."
            
    except Exception as e:
        # Log the error or handle it more gracefully
        print(f"Error generating testing suite: {e}")
        return f"# Error generating test suite: {e}"
