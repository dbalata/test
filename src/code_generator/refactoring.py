"""
Code refactoring functionality.
"""

from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


def refactor_code(
    llm,
    original_code: str,
    refactoring_goals: str,
    language: str = "python"
) -> Dict[str, Any]:
    """
    Refactor existing code based on specified goals.
    
    Args:
        llm: Language model instance
        original_code: Source code to refactor
        refactoring_goals: Description of desired improvements
        language: Programming language of the code
        
    Returns:
        Dict containing refactored code and explanation
    """
    prompt = PromptTemplate(
        input_variables=["original_code", "refactoring_goals", "language"],
        template="""Refactor the following {language} code with these goals:
        {refactoring_goals}
        
        Original code:
        ```{language}
        {original_code}
        ```
        
        Provide:
        1. Refactored code with improvements
        2. Explanation of changes made
        3. Any performance or readability benefits
        
        Format your response with markdown code blocks."""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(
        original_code=original_code,
        refactoring_goals=refactoring_goals,
        language=language
    )
    return result
