import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

from src.document_processor import DocumentProcessor
from src.qa_system import QASystem
from src.agents import create_research_agent
from src.utils import setup_logging

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
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please set your OPENAI_API_KEY in the .env file")
            st.stop()
        
        st.success("✅ OpenAI API Key configured")
        
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
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    st.chat_message("user").write(message.content)
                elif isinstance(message, AIMessage):
                    st.chat_message("assistant").write(message.content)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.documents_loaded:
                st.warning("Please upload and process documents first!")
            else:
                # Add user message to chat history
                user_message = HumanMessage(content=prompt)
                st.session_state.chat_history.append(user_message)
                
                # Display user message
                st.chat_message("user").write(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.qa_system.ask_question(prompt)
                            st.write(response["answer"])
                            
                            # Show sources if available
                            if "sources" in response and response["sources"]:
                                with st.expander("📚 Sources"):
                                    for i, source in enumerate(response["sources"]):
                                        st.write(f"**Source {i+1}:**")
                                        st.write(source)
                                        st.write("---")
                            
                            # Add AI message to chat history
                            ai_message = AIMessage(content=response["answer"])
                            st.session_state.chat_history.append(ai_message)
                            
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            logger.error(error_msg)
    
    with col2:
        st.subheader("🔍 Research Agent")
        
        if st.button("🌐 Enable Research Agent"):
            if not st.session_state.documents_loaded:
                st.warning("Please process documents first!")
            else:
                try:
                    research_agent = create_research_agent()
                    st.success("✅ Research agent enabled!")
                    st.info("You can now ask questions that require web research in addition to document analysis.")
                except Exception as e:
                    st.error(f"Error creating research agent: {str(e)}")
        
        # System information
        st.subheader("ℹ️ System Info")
        if st.session_state.documents_loaded:
            st.success("📄 Documents: Loaded")
            if st.session_state.qa_system:
                doc_count = len(st.session_state.qa_system.vector_store.get()['ids'])
                st.info(f"📊 Document chunks: {doc_count}")
        else:
            st.warning("📄 Documents: Not loaded")
        
        st.info(f"💭 Memory: {len(st.session_state.chat_history)} messages")

if __name__ == "__main__":
    main()
