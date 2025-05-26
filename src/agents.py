"""
Advanced agents with specialized tools and capabilities.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import SerpAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import PromptTemplate
from langchain.agents.agent_toolkits.base import BaseToolkit
from .openrouter_utils import get_openrouter_llm
import requests
import json

# Import the new code generator
from .code_generator import CodeGenerator


class WebSearchTool:
    """Enhanced web search tool with multiple search engines."""
    
    def __init__(self):
        self.search_engine = None
        self._setup_search()
    
    def _setup_search(self):
        """Setup the best available search engine."""
        if os.getenv("SERPAPI_API_KEY"):
            self.search_engine = SerpAPIWrapper()
        else:
            self.search_engine = DuckDuckGoSearchRun()
    
    def search(self, query: str) -> str:
        """Perform web search."""
        try:
            return self.search_engine.run(query)
        except Exception as e:
            return f"Search error: {str(e)}"


class DocumentAnalysisTool:
    """Tool for analyzing documents in the vector store."""
    
    def __init__(self, qa_system):
        self.qa_system = qa_system
    
    def analyze_document_sentiment(self, query: str) -> str:
        """Analyze sentiment of documents related to query."""
        try:
            docs = self.qa_system.get_similar_documents(query, k=3)
            
            if not docs:
                return "No relevant documents found for sentiment analysis."
            
            # Combine document contents
            combined_text = "\n".join([doc.page_content for doc in docs])
            
            # Use LLM for sentiment analysis
            sentiment_prompt = f"""
            Analyze the sentiment of the following text and provide:
            1. Overall sentiment (positive, negative, neutral)
            2. Key emotional indicators
            3. Confidence level
            
            Text: {combined_text[:1000]}...
            """
            
            result = self.qa_system.llm.predict(sentiment_prompt)
            return result
            
        except Exception as e:
            return f"Error in sentiment analysis: {str(e)}"
    
    def extract_key_topics(self, query: str) -> str:
        """Extract key topics from documents."""
        try:
            docs = self.qa_system.get_similar_documents(query, k=5)
            
            if not docs:
                return "No relevant documents found for topic extraction."
            
            combined_text = "\n".join([doc.page_content for doc in docs])
            
            topic_prompt = f"""
            Extract the main topics and themes from the following text:
            
            {combined_text[:1500]}...
            
            Provide:
            1. Top 5 main topics
            2. Key concepts for each topic
            3. Relationships between topics
            """
            
            result = self.qa_system.llm.predict(topic_prompt)
            return result
            
        except Exception as e:
            return f"Error in topic extraction: {str(e)}"


class CodeAnalysisTool:
    """Tool for analyzing code in documents."""
    
    def __init__(self):
        self.python_repl = PythonREPLTool()
    
    def analyze_code_snippet(self, code: str) -> str:
        """Analyze a code snippet."""
        try:
            analysis_code = f"""
# Code to analyze:
{code}

# Analysis
import ast
import sys
from io import StringIO

def analyze_code():
    code_to_analyze = '''{code}'''
    
    try:
        tree = ast.parse(code_to_analyze)
        
        # Count different node types
        node_counts = {{}}
        for node in ast.walk(tree):
            node_type = type(node).__name__
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        print("Code Analysis Results:")
        print(f"Total AST nodes: {{sum(node_counts.values())}}")
        print("Node type distribution:")
        for node_type, count in sorted(node_counts.items()):
            print(f"  {{node_type}}: {{count}}")
        
        # Check for common patterns
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if functions:
            print(f"Functions found: {{', '.join(functions)}}")
        if classes:
            print(f"Classes found: {{', '.join(classes)}}")
        
    except SyntaxError as e:
        print(f"Syntax error in code: {{e}}")
    except Exception as e:
        print(f"Error analyzing code: {{e}}")

analyze_code()
"""
            
            return self.python_repl.run(analysis_code)
            
        except Exception as e:
            return f"Error analyzing code: {str(e)}"


class CodeGenerationTool:
    """Advanced code generation tool with templates and AI assistance."""
    
    def __init__(self):
        self.generator = CodeGenerator()
    
    def generate_code_from_description(self, description: str) -> str:
        """Generate code from natural language description."""
        try:
            # Extract language and framework hints from description
            language = "python"  # default
            framework = None
            
            if "javascript" in description.lower() or "js" in description.lower():
                language = "javascript"
            elif "react" in description.lower():
                language = "javascript"
                framework = "react"
            elif "flask" in description.lower():
                framework = "flask"
            elif "fastapi" in description.lower():
                framework = "fastapi"
            
            result = self.generator.generate_with_ai(description, language, framework)
            
            if result['success']:
                output = f"Generated {language} code"
                if framework:
                    output += f" using {framework}"
                output += f":\n\n{result['result']['explanation']}\n\n"
                
                for block in result['result']['code_blocks']:
                    output += f"```{block['language']}\n{block['code']}\n```\n\n"
                
                if result['result']['dependencies']:
                    output += "Dependencies:\n"
                    for dep in result['result']['dependencies']:
                        output += f"- {dep}\n"
                
                return output
            else:
                return f"Error generating code: {result['error']}"
                
        except Exception as e:
            return f"Error in code generation: {str(e)}"
    
    def generate_from_template(self, template_info: str) -> str:
        """Generate code using predefined templates."""
        try:
            # Parse template info - expected format: "template_name:param1=value1,param2=value2"
            if ":" not in template_info:
                return f"Available templates: {', '.join(self.generator.get_available_templates().keys())}"
            
            template_name, params_str = template_info.split(":", 1)
            
            # Parse parameters
            params = {}
            if params_str.strip():
                for param in params_str.split(","):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        params[key.strip()] = value.strip()
            
            result = self.generator.generate_from_template(template_name.strip(), **params)
            
            if result['success']:
                return f"Generated code using {template_name} template:\n\n```\n{result['code']}\n```"
            else:
                return f"Error: {result['error']}"
                
        except Exception as e:
            return f"Error in template generation: {str(e)}"
    
    def generate_api_client(self, api_spec: str) -> str:
        """Generate API client code."""
        try:
            # Parse API specification
            lines = api_spec.split('\n')
            description = lines[0] if lines else "API Client"
            base_url = "https://api.example.com"
            endpoints = []
            
            # Simple parsing - in production would use proper API spec parser
            for line in lines[1:]:
                if line.strip().startswith("URL:"):
                    base_url = line.split(":", 1)[1].strip()
                elif line.strip().startswith("GET") or line.strip().startswith("POST"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        endpoints.append({
                            "method": parts[0],
                            "path": parts[1],
                            "description": " ".join(parts[2:]) if len(parts) > 2 else ""
                        })
            
            result = self.generator.generate_api_client(description, base_url, endpoints)
            
            if result['success']:
                output = f"Generated API client for: {description}\n\n"
                output += result['client_code']['explanation'] + "\n\n"
                
                for block in result['client_code']['code_blocks']:
                    output += f"```{block['language']}\n{block['code']}\n```\n\n"
                
                return output
            else:
                return f"Error generating API client: {result['error']}"
                
        except Exception as e:
            return f"Error in API client generation: {str(e)}"
    
    def explain_code(self, code: str) -> str:
        """Explain existing code."""
        try:
            result = self.generator.explain_code(code, detail_level="detailed")
            
            if result['success']:
                return f"Code Explanation:\n\n{result['explanation']}"
            else:
                return f"Error explaining code: {result['error']}"
                
        except Exception as e:
            return f"Error in code explanation: {str(e)}"


def create_research_agent(qa_system=None) -> AgentExecutor:
    """Create a research agent with multiple tools."""
    
    # Initialize tools
    web_search = WebSearchTool()
    code_generator = CodeGenerationTool()
    
    tools = [
        Tool(
            name="web_search",
            description="Search the web for current information. Use this when you need up-to-date information not in the documents.",
            func=web_search.search
        ),
        Tool(
            name="python_calculator",
            description="Execute Python code for calculations, data analysis, or code generation.",
            func=code_generator.python_repl.run
        ),
        Tool(
            name="generate_code",
            description="Generate code from natural language description. Provide a clear description of what you want to build.",
            func=code_generator.generate_code_from_description
        ),
        Tool(
            name="code_template",
            description="Generate code using predefined templates. Format: 'template_name:param1=value1,param2=value2'. Call without parameters to see available templates.",
            func=code_generator.generate_from_template
        ),
        Tool(
            name="generate_api_client",
            description="Generate API client code. Provide API description with endpoints in format: 'Description\\nURL: base_url\\nGET /endpoint1\\nPOST /endpoint2'",
            func=code_generator.generate_api_client
        ),
        Tool(
            name="explain_code",
            description="Explain existing code with detailed analysis. Provide the code you want explained.",
            func=code_generator.explain_code
        )
    ]
    
    # Add document analysis tools if qa_system is available
    if qa_system:
        doc_analyzer = DocumentAnalysisTool(qa_system)
        
        tools.extend([
            Tool(
                name="document_sentiment",
                description="Analyze sentiment of documents related to a query.",
                func=doc_analyzer.analyze_document_sentiment
            ),
            Tool(
                name="extract_topics",
                description="Extract key topics and themes from documents.",
                func=doc_analyzer.extract_key_topics
            )
        ])
    
    # Create the agent
    llm = get_openrouter_llm(
        model_name="openai/gpt-3.5-turbo",
        temperature=0.2
    )
    
    # Define the agent prompt
    agent_prompt = PromptTemplate.from_template("""
    You are a research assistant with access to various tools including advanced code generation capabilities. Your goal is to provide comprehensive, accurate answers by combining information from documents, web searches, and code generation when necessary.

    Available tools:
    {tools}

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    When asked to generate code:
    1. Use generate_code for custom requirements
    2. Use code_template for common patterns
    3. Use generate_api_client for API-related tasks
    4. Use explain_code to understand existing code
    5. Use python_calculator for testing or running code

    Question: {input}
    Thought: {agent_scratchpad}
    """)
    
    # Create the agent
    agent = create_react_agent(llm, tools, agent_prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        early_stopping_method="generate"
    )
    
    return agent_executor


def create_code_generation_agent() -> AgentExecutor:
    """Create a specialized agent focused on code generation tasks."""
    
    code_generator = CodeGenerationTool()
    code_analyzer = CodeAnalysisTool()
    
    # Wrapper functions to ensure consistent return types
    def wrap_tool(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if isinstance(result, dict) and 'success' in result:
                    if result['success']:
                        return str(result.get('result', result.get('explanation', 'Success'))) if 'result' in result or 'explanation' in result else 'Success'
                    else:
                        return f"Error: {result.get('error', 'Unknown error')}"
                return str(result) if result is not None else 'No output'
            except Exception as e:
                return f"Error executing tool: {str(e)}"
        return wrapper
    
    tools = [
        Tool(
            name="generate_code",
            description="Generate code from natural language description. Provide a clear description of what you want to build.",
            func=wrap_tool(code_generator.generate_code_from_description)
        ),
        Tool(
            name="code_template",
            description="Generate code using predefined templates. Provide template name and required variables.",
            func=wrap_tool(code_generator.generate_from_template)
        ),
        Tool(
            name="generate_api_client",
            description="Generate API client code. Provide API description, base URL, and endpoints as a JSON string.",
            func=wrap_tool(code_generator.generate_api_client)
        ),
        Tool(
            name="explain_code",
            description="Explain what a piece of code does. Provide the code and optionally the detail level (basic, detailed, or advanced).",
            func=wrap_tool(code_generator.explain_code)
        ),
        Tool(
            name="analyze_code",
            description="Analyze a code snippet. Provide the code to analyze.",
            func=wrap_tool(code_analyzer.analyze_code_snippet)
        )
    ]
    
    llm = get_openrouter_llm(
        model_name="openai/gpt-3.5-turbo",
        temperature=0.1
    )
    
    agent_prompt = PromptTemplate.from_template("""
You are a specialized code generation assistant. You excel at creating high-quality, production-ready code in various programming languages and frameworks.

Available tools:
{tools}

Your capabilities include:
- Generating custom code from descriptions
- Using predefined templates for common patterns
- Creating API clients and integrations
- Explaining and analyzing existing code

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}
""")
    
    agent = create_react_agent(llm, tools, agent_prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,  # Reduced from 10 to prevent long-running operations
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )


class MultiAgentSystem:
    """System for managing multiple specialized agents."""
    
    def __init__(self, qa_system=None):
        self.qa_system = qa_system
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize different types of agents."""
        try:
            # Research agent
            self.agents['research'] = create_research_agent(self.qa_system)
            
            # Code generation agent
            self.agents['code_generation'] = create_code_generation_agent()
            
            # Add more specialized agents here as needed
            # self.agents['data_analysis'] = create_data_analysis_agent()
            # self.agents['code_review'] = create_code_review_agent()
            
        except Exception as e:
            print(f"Error initializing agents: {str(e)}")
    
    def route_query(self, query: str) -> str:
        """Route query to the most appropriate agent."""
        # Enhanced routing logic for code generation
        query_lower = query.lower()
        
        # Code generation keywords
        code_keywords = [
            'generate', 'create', 'build', 'write', 'code', 'function', 'class', 
            'api', 'client', 'template', 'script', 'program', 'implement',
            'flask', 'react', 'python', 'javascript', 'sql', 'database'
        ]
        
        if any(keyword in query_lower for keyword in code_keywords):
            return 'code_generation'
        elif any(keyword in query_lower for keyword in ['search', 'web', 'current', 'latest', 'news']):
            return 'research'
        elif any(keyword in query_lower for keyword in ['analyze', 'sentiment', 'topics']):
            return 'research'  # Research agent has analysis tools
        else:
            return 'research'  # Default to research agent
    
    def process_query(self, query: str, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Process a query using the appropriate agent."""
        if agent_type is None:
            agent_type = self.route_query(query)
        
        if agent_type not in self.agents:
            return {
                "error": f"Agent type '{agent_type}' not available",
                "available_agents": list(self.agents.keys())
            }
        
        try:
            agent = self.agents[agent_type]
            result = agent.run(query)
            
            return {
                "agent_used": agent_type,
                "result": result,
                "query": query
            }
            
        except Exception as e:
            return {
                "error": f"Error processing query with {agent_type} agent: {str(e)}",
                "query": query
            }
