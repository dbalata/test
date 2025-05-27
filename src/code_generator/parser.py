"""
Parser for code generation outputs.
Handles extraction of code blocks, explanations, and metadata from LLM responses.
"""

import re
from typing import Dict, Any
from langchain_core.output_parsers import BaseOutputParser


class CodeOutputParser(BaseOutputParser):
    """Custom parser for extracting code from LLM responses."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse the output to extract code, explanation, and metadata.
        
        Args:
            text: Raw text output from LLM
            
        Returns:
            Dict containing parsed sections (explanation, code_blocks, etc.)
        """
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
        deps_match = re.search(
            r'(?:dependencies|requirements|install):(.*?)(?:\n\n|\n```|$)', 
            text, 
            re.IGNORECASE | re.DOTALL
        )
        if deps_match:
            deps_text = deps_match.group(1).strip()
            sections['dependencies'] = [dep.strip() for dep in deps_text.split('\n') if dep.strip()]
        
        return sections
