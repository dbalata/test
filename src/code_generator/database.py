"""
Database schema generation functionality.
"""

from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


def generate_database_schema(
    llm,
    requirements: str,
    database_type: str = "postgresql"
) -> Dict[str, Any]:
    """
    Generate database schema based on requirements.
    
    Args:
        llm: Language model instance
        requirements: Natural language description of database requirements
        database_type: Type of database (e.g., 'postgresql', 'mysql', 'mongodb')
        
    Returns:
        Dict containing generated schema and metadata
    """
    prompt = PromptTemplate(
        input_variables=["requirements", "database_type"],
        template="""Generate a {database_type} database schema based on the following requirements:
        
        {requirements}
        
        Include:
        1. Table definitions with appropriate data types and constraints
        2. Primary and foreign key relationships
        3. Indexes for frequently queried fields
        4. Any necessary views or stored procedures
        5. Sample data insertion queries
        
        Format your response with markdown code blocks."""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(requirements=requirements, database_type=database_type)
    return result
