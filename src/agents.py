"""
Advanced agents with specialized tools and capabilities.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import SerpAPIWrapper
from langchain.tools.python.tool import PythonREPLTool
from langchain.prompts import PromptTemplate
from langchain.schema import BaseToolkit
import requests
import json


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


def create_research_agent(qa_system=None) -> AgentExecutor:
    """Create a research agent with multiple tools."""
    
    # Initialize tools
    web_search = WebSearchTool()
    
    tools = [
        Tool(
            name="web_search",
            description="Search the web for current information. Use this when you need up-to-date information not in the documents.",
            func=web_search.search
        ),
        Tool(
            name="python_calculator",
            description="Execute Python code for calculations, data analysis, or code generation.",
            func=PythonREPLTool().run
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
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
    
    # Define the agent prompt
    agent_prompt = PromptTemplate.from_template("""
    You are a research assistant with access to various tools. Your goal is to provide comprehensive, accurate answers by combining information from documents and web searches when necessary.

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
            
            # Add more specialized agents here as needed
            # self.agents['data_analysis'] = create_data_analysis_agent()
            # self.agents['code_review'] = create_code_review_agent()
            
        except Exception as e:
            print(f"Error initializing agents: {str(e)}")
    
    def route_query(self, query: str) -> str:
        """Route query to the most appropriate agent."""
        # Simple routing logic - can be enhanced with ML classification
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['search', 'web', 'current', 'latest', 'news']):
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
