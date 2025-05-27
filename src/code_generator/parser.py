"""
Parser for code generation outputs.
Handles extraction of code blocks, explanations, and metadata from LLM responses.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict

from .models.generation import CodeBlock, GenerationResult


class ParserError(Exception):
    """Base exception for parser-related errors."""
    pass


class CodeParser:
    """Parser for extracting structured data from LLM responses."""
    
    # Regex patterns for parsing
    CODE_BLOCK_PATTERN = r'```(\w+)?\n(.*?)\n```'
    DEPENDENCIES_PATTERN = r'(?:dependencies|requirements|install):(.*?)(?:\n\n|\n```|$)'
    USAGE_EXAMPLES_PATTERN = r'(?:usage|example|examples):(.*?)(?:\n\n|\n```|$)'
    
    def parse(self, text: str) -> GenerationResult:
        """
        Parse LLM response into structured data.
        
        Args:
            text: Raw text output from LLM
            
        Returns:
            GenerationResult containing parsed data
            
        Raises:
            ParserError: If parsing fails
        """
        if not text or not isinstance(text, str):
            raise ParserError("Input text must be a non-empty string")
            
        try:
            code_blocks = self._extract_code_blocks(text)
            explanation = self._extract_explanation(text)
            dependencies = self._extract_dependencies(text)
            usage_examples = self._extract_usage_examples(text)
            
            return GenerationResult(
                explanation=explanation,
                code_blocks=code_blocks,
                dependencies=dependencies,
                usage_examples=usage_examples,
                metadata={
                    'raw_text': text,
                    'has_code': len(code_blocks) > 0
                }
            )
        except Exception as e:
            raise ParserError(f"Failed to parse LLM response: {str(e)}")
    
    def _extract_code_blocks(self, text: str) -> List[CodeBlock]:
        """Extract code blocks from text with their languages."""
        blocks = re.findall(self.CODE_BLOCK_PATTERN, text, re.DOTALL)
        return [
            CodeBlock(
                language=lang.lower() if lang else 'text',
                code=code.strip()
            )
            for lang, code in blocks
        ]
    
    def _extract_explanation(self, text: str) -> str:
        """Extract explanation text before the first code block."""
        # Find the first code block
        first_code = re.search(r'```', text)
        if not first_code:
            return text.strip()
            
        # Get everything before the first code block
        explanation = text[:first_code.start()].strip()
        
        # Remove any section headers like "Explanation:" or "Answer:"
        explanation = re.sub(r'^(Explanation|Answer|Code|Solution)[:\s]*\n*', 
                           '', 
                           explanation, 
                           flags=re.IGNORECASE)
        
        return explanation.strip()
    
    def _extract_dependencies(self, text: str) -> List[str]:
        """Extract dependencies from text."""
        match = re.search(
            self.DEPENDENCIES_PATTERN, 
            text, 
            re.IGNORECASE | re.DOTALL
        )
        if not match:
            return []
            
        deps_text = match.group(1).strip()
        return [
            dep.strip() 
            for dep in deps_text.split('\n') 
            if dep.strip() and not dep.strip().startswith('```')
        ]
    
    def _extract_usage_examples(self, text: str) -> List[str]:
        """Extract usage examples from text."""
        match = re.search(
            self.USAGE_EXAMPLES_PATTERN,
            text,
            re.IGNORECASE | re.DOTALL
        )
        if not match:
            return []
            
        examples_text = match.group(1).strip()
        return [
            ex.strip()
            for ex in re.split(r'\n\s*\d+[\.\)]\s*', examples_text)
            if ex.strip()
        ]


# Default parser instance
parse_code = CodeParser().parse
