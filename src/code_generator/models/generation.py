"""
Data models for code generation results and templates.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class CodeBlock:
    """Represents a block of code with its language."""
    language: str
    code: str


@dataclass
class GenerationResult:
    """Container for code generation results."""
    explanation: str
    code_blocks: List[CodeBlock] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    usage_examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateConfig:
    """Configuration for code templates."""
    name: str
    description: str
    variables: List[str]
    template: str
