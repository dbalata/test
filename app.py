import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

from src.document_processor import DocumentProcessor
from src.qa_system import QASystem
from src.agents import create_research_agent, create_code_generation_agent, MultiAgentSystem
from src.code_generator import CodeGenerator
from src.utils import setup_logging
from src.openrouter_utils import get_openrouter_llm

# Load environment variables
load_dotenv()

# Setup logging
logger = setup_logging()

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=10, return_messages=True
        )
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'code_generator' not in st.session_state:
        st.session_state.code_generator = CodeGenerator()
    if 'multi_agent_system' not in st.session_state:
        st.session_state.multi_agent_system = None

def render_code_generation_section():
    """Render the code generation section in the sidebar."""
    st.subheader("🧩 Code Generation")
    
    # Code generation mode selection
    gen_mode = st.selectbox(
        "Generation Mode",
        ["AI Description", "Template", "API Client", "Explain Code"],
        help="Choose how you want to generate code"
    )
    
    if gen_mode == "AI Description":
        description = st.text_area(
            "Describe what you want to build:",
            placeholder="Create a REST API endpoint that handles user authentication with JWT tokens"
        )
        language = st.selectbox("Language", ["python", "javascript", "sql", "other"])
        framework = st.text_input("Framework (optional)", placeholder="flask, react, fastapi")
        
        if st.button("🚀 Generate Code"):
            if description:
                with st.spinner("Generating code..."):
                    try:
                        result = st.session_state.code_generator.generate_with_ai(
                            description, language, framework or None
                        )
                        if result['success']:
                            st.success("Code generated successfully!")
                            
                            # Display explanation
                            if result['result']['explanation']:
                                st.write("**Explanation:**")
                                st.write(result['result']['explanation'])
                            
                            # Display code blocks
                            for i, block in enumerate(result['result']['code_blocks']):
                                st.write(f"**Code ({block['language']}):**")
                                st.code(block['code'], language=block['language'])
                            
                            # Display dependencies
                            if result['result']['dependencies']:
                                st.write("**Dependencies:**")
                                for dep in result['result']['dependencies']:
                                    st.write(f"- {dep}")
                        else:
                            st.error(f"Error: {result['error']}")
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
            else:
                st.warning("Please provide a description")
    
    elif gen_mode == "Template":
        # Show available templates
        templates = st.session_state.code_generator.get_available_templates()
        template_name = st.selectbox("Select Template", list(templates.keys()))
        
        if template_name:
            st.write(f"**Description:** {templates[template_name]}")
            
            # Template-specific parameters
            if template_name == "python_class":
                class_name = st.text_input("Class Name", "MyClass")
                description = st.text_input("Description", "A sample class")
                
                if st.button("Generate from Template"):
                    try:
                        result = st.session_state.code_generator.generate_from_template(
                            template_name,
                            class_name=class_name,
                            description=description,
                            init_params="",
                            init_body="pass",
                            methods="def sample_method(self):\n        pass"
                        )
                        if result['success']:
                            st.code(result['code'], language='python')
                        else:
                            st.error(result['error'])
                    except Exception as e:
                        st.error(f"Template generation failed: {str(e)}")
            
            elif template_name == "flask_api":
                endpoint = st.text_input("Endpoint", "users")
                method = st.selectbox("Method", ["GET", "POST", "PUT", "DELETE"])
                function_name = st.text_input("Function Name", "handle_users")
                
                if st.button("Generate from Template"):
                    try:
                        result = st.session_state.code_generator.generate_from_template(
                            template_name,
                            endpoint=endpoint,
                            method=method,
                            function_name=function_name,
                            description=f"Handle {method} requests for {endpoint}",
                            function_body="    # Add your logic here\n    pass",
                            return_value="{'status': 'success'}"
                        )
                        if result['success']:
                            st.code(result['code'], language='python')
                        else:
                            st.error(result['error'])
                    except Exception as e:
                        st.error(f"Template generation failed: {str(e)}")
    
    elif gen_mode == "API Client":
        api_description = st.text_input("API Description", "User Management API")
        base_url = st.text_input("Base URL", "https://api.example.com")
        endpoints_text = st.text_area(
            "Endpoints (one per line)",
            "GET /users - Get all users\nPOST /users - Create user\nGET /users/{id} - Get user by ID"
        )
        
        if st.button("Generate API Client"):
            if api_description and base_url and endpoints_text:
                with st.spinner("Generating API client..."):
                    try:
                        # Parse endpoints
                        endpoints = []
                        for line in endpoints_text.split('\n'):
                            if line.strip() and ' ' in line:
                                parts = line.split(' ', 2)
                                if len(parts) >= 2:
                                    endpoints.append({
                                        'method': parts[0],
                                        'path': parts[1],
                                        'description': parts[2] if len(parts) > 2 else ''
                                    })
                        
                        result = st.session_state.code_generator.generate_api_client(
                            api_description, base_url, endpoints
                        )
                        
                        if result['success']:
                            st.success("API client generated!")
                            
                            # Display explanation
                            st.write("**Explanation:**")
                            st.write(result['client_code']['explanation'])
                            
                            # Display code
                            for block in result['client_code']['code_blocks']:
                                st.code(block['code'], language=block['language'])
                        else:
                            st.error(f"Error: {result['error']}")
                    except Exception as e:
                        st.error(f"API client generation failed: {str(e)}")
            else:
                st.warning("Please fill in all fields")
    
    elif gen_mode == "Explain Code":
        code_input = st.text_area(
            "Paste your code here:",
            placeholder="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        )
        detail_level = st.selectbox("Detail Level", ["brief", "detailed", "expert"])
        
        if st.button("Explain Code"):
            if code_input:
                with st.spinner("Analyzing code..."):
                    try:
                        result = st.session_state.code_generator.explain_code(
                            code_input, detail_level
                        )
                        if result['success']:
                            st.write("**Code Explanation:**")
                            st.write(result['explanation'])
                        else:
                            st.error(f"Error: {result['error']}")
                    except Exception as e:
                        st.error(f"Code explanation failed: {str(e)}")
            else:
                st.warning("Please provide code to explain")

def main():
    st.set_page_config(
        page_title="Advanced LangChain Document Q&A System",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🤖 Advanced LangChain Document Q&A System")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key check
        if not os.getenv("OPENROUTER_API_KEY"):
            st.error("Please set your OPENROUTER_API_KEY in the .env file")
            st.stop()
        
        st.success("✅ OpenRouter API Key configured")
        
        # Document upload section
        st.subheader("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        # URL input for web scraping
        url_input = st.text_input("Enter URL to scrape:")
        
        # Process documents button
        if st.button("🔄 Process Documents"):
            if uploaded_files or url_input:
                with st.spinner("Processing documents..."):
                    try:
                        processor = DocumentProcessor()
                        
                        # Process uploaded files
                        if uploaded_files:
                            for file in uploaded_files:
                                processor.add_file(file)
                        
                        # Process URL
                        if url_input:
                            processor.add_url(url_input)
                        
                        # Create vector store
                        vector_store = processor.create_vector_store()
                        
                        # Initialize QA system
                        st.session_state.qa_system = QASystem(
                            vector_store=vector_store,
                            memory=st.session_state.memory
                        )
                        
                        # Initialize multi-agent system
                        st.session_state.multi_agent_system = MultiAgentSystem(
                            qa_system=st.session_state.qa_system
                        )
                        
                        st.session_state.documents_loaded = True
                        st.success("✅ Documents processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        logger.error(f"Document processing error: {str(e)}")
            else:
                st.warning("Please upload files or enter a URL")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Code Generation Section
        render_code_generation_section()
    
    # Main content area
    st.subheader("💬 Chat Interface")
    
    # Agent selection
    agent_type = st.selectbox(
        "Select Agent Type",
        ["auto", "research", "code_generation"],
        help="Choose which agent to use, or let the system decide automatically"
    )
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            elif isinstance(message, AIMessage):
                st.chat_message("assistant").write(message.content)
    
    # Chat input - must be at the bottom level of the app, not inside columns or other containers
    if prompt := st.chat_input("Ask a question about your documents or request code generation..."):
            # Add user message to chat history
            user_message = HumanMessage(content=prompt)
            st.session_state.chat_history.append(user_message)
            
            # Display user message
            st.chat_message("user").write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Use multi-agent system if available and documents are loaded
                        if st.session_state.multi_agent_system and agent_type != "auto":
                            # Use specific agent
                            response = st.session_state.multi_agent_system.process_query(
                                prompt, agent_type if agent_type != "auto" else None
                            )
                            
                            if "error" in response:
                                st.error(response["error"])
                                answer = f"Error: {response['error']}"
                            else:
                                st.write(f"*Agent used: {response['agent_used']}*")
                                st.write(response["result"])
                                answer = response["result"]
                        
                        elif st.session_state.documents_loaded and st.session_state.qa_system:
                            # Use regular QA system
                            response = st.session_state.qa_system.ask_question(prompt)
                            st.write(response["answer"])
                            
                            # Show sources if available
                            if "sources" in response and response["sources"]:
                                with st.expander("📚 Sources"):
                                    for i, source in enumerate(response["sources"]):
                                        st.write(f"**Source {i+1}:**")
                                        st.write(source)
                                        st.write("---")
                            
                            answer = response["answer"]
                        
                        else:
                            # No documents loaded, use code generation agent if it's a code-related query
                            if any(keyword in prompt.lower() for keyword in ['generate', 'create', 'code', 'build', 'write']):
                                try:
                                    code_agent = create_code_generation_agent()
                                    result = code_agent.run(prompt)
                                    st.write("*Using code generation agent*")
                                    st.write(result)
                                    answer = result
                                except Exception as e:
                                    st.error(f"Code generation error: {str(e)}")
                                    answer = f"Error: {str(e)}"
                            else:
                                st.warning("Please upload and process documents first, or ask a code generation question!")
                                answer = "Please upload documents first or ask about code generation."
                        
                        # Add AI message to chat history
                        ai_message = AIMessage(content=answer)
                        st.session_state.chat_history.append(ai_message)
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        logger.error(error_msg)
    
    # System status section
    st.markdown("---")
    st.subheader("🔍 System Status")
    
    # System information in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ℹ️ System Info")
        if st.session_state.documents_loaded:
            st.success("📄 Documents: Loaded")
            if st.session_state.qa_system and hasattr(st.session_state.qa_system, 'vector_store'):
                try:
                    doc_count = len(st.session_state.qa_system.vector_store.get()['ids'])
                    st.info(f"📊 Document chunks: {doc_count}")
                except:
                    st.info("📊 Document chunks: Information not available")
        else:
            st.warning("📄 Documents: Not loaded")
    
    with col2:
        st.subheader("🤖 Agent Status")
        if st.session_state.multi_agent_system:
            st.success("Multi-Agent System: Active")
            available_agents = list(st.session_state.multi_agent_system.agents.keys())
            st.info(f"Available agents: {', '.join(available_agents)}")
        else:
            st.warning("Multi-Agent System: Not initialized")
        
        st.info(f"💭 Memory: {len(st.session_state.chat_history)} messages")
    
    # Quick actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    action_col1, action_col2 = st.columns(2)
    
    with action_col1:
        if st.button("🧪 Test Code Generation", use_container_width=True):
            try:
                print("\n[INFO] Initializing code generation agent...")
                code_agent = create_code_generation_agent()
                print("[INFO] Agent initialized. Sending request...")
                result = code_agent.invoke({"input": "Create a simple Python function to calculate fibonacci numbers"})
                print("[INFO] Request completed successfully")
                st.code(result)
            except Exception as e:
                import traceback
                error_msg = f"Test failed: {str(e)}\n{traceback.format_exc()}"
                print(f"\n[ERROR] {error_msg}")
                st.error(f"Test failed: {str(e)}")
    
    with action_col2:
        if st.button("📊 Show Templates", use_container_width=True):
            try:
                print("\n[INFO] Loading available templates...")
                templates = st.session_state.code_generator.get_available_templates()
                if templates:
                    print(f"[INFO] Found {len(templates)} templates")
                    for name, desc in templates.items():
                        st.write(f"**{name}**: {desc}")
                else:
                    print("[INFO] No templates available")
                    st.info("No templates available")
            except Exception as e:
                import traceback
                error_msg = f"Failed to load templates: {str(e)}\n{traceback.format_exc()}"
                print(f"\n[ERROR] {error_msg}")
                st.error(f"Failed to load templates: {str(e)}")

if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("Starting Streamlit application...")
        print("="*50 + "\n")
        main()
    except Exception as e:
        import traceback
        error_msg = f"Fatal error in main: {str(e)}\n{traceback.format_exc()}"
        print(f"\n[FATAL] {error_msg}")
        raise
