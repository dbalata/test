"""
Question-Answering system using Retrieval-Augmented Generation (RAG).
"""

from typing import Dict, List, Any
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Import our OpenRouter utilities
from .openrouter_utils import get_openrouter_llm


class CustomQAPromptTemplate:
    """Custom prompt templates for different types of questions."""
    
    @staticmethod
    def get_general_qa_prompt():
        """General Q&A prompt template."""
        template = """You are an AI assistant that answers questions based on the provided context from documents. 
        Use the following pieces of context to answer the human's question at the end. 
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Human: {question}
        
        Assistant: I'll answer your question based on the provided documents. """
        
        return PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
    
    @staticmethod
    def get_analytical_prompt():
        """Analytical prompt for deeper analysis."""
        template = """You are an expert analyst. Analyze the provided context to answer the human's question.
        Provide detailed insights, identify patterns, and make connections between different pieces of information.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Human: {question}
        
        Assistant: Based on my analysis of the documents: """
        
        return PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )


class QASystem:
    """Main Question-Answering system with RAG capabilities."""
    
    def __init__(self, vector_store: Chroma, memory: ConversationBufferWindowMemory):
        self.vector_store = vector_store
        self.memory = memory
        self.llm = get_openrouter_llm(
            model_name="openai/gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000
        )
        
        # Create retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Initialize the conversational chain
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup the conversational retrieval chain."""
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
    
    def ask_question(self, question: str, analysis_mode: bool = False) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: The question to ask
            analysis_mode: Whether to use analytical prompt
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        try:
            # Get the response from the chain
            result = self.qa_chain({"question": question})
            
            # Process source documents
            sources = []
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
            
            # Enhance answer with additional analysis if requested
            answer = result["answer"]
            if analysis_mode:
                answer = self._enhance_analysis(answer, sources)
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "confidence": self._calculate_confidence(result)
            }
            
        except Exception as e:
            raise Exception(f"Error processing question: {str(e)}")
    
    def _enhance_analysis(self, base_answer: str, sources: List[Dict]) -> str:
        """Enhance the answer with additional analytical insights."""
        if len(sources) > 1:
            analysis_prompt = f"""
            Based on the answer: {base_answer}
            
            And considering {len(sources)} different sources, provide additional insights about:
            1. Key patterns or themes
            2. Potential contradictions or agreements
            3. Areas that might need further investigation
            
            Keep it concise but insightful.
            """
            
            try:
                enhanced = self.llm.predict(analysis_prompt)
                return f"{base_answer}\n\n**Additional Analysis:**\n{enhanced}"
            except:
                return base_answer
        
        return base_answer
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate a confidence score for the answer."""
        # Simple heuristic based on number of sources and answer length
        num_sources = len(result.get("source_documents", []))
        answer_length = len(result.get("answer", ""))
        
        # Base confidence on sources (more sources = higher confidence)
        source_confidence = min(num_sources * 0.2, 1.0)
        
        # Adjust based on answer length (very short or very long answers might be less confident)
        if 50 <= answer_length <= 500:
            length_confidence = 1.0
        elif answer_length < 50:
            length_confidence = 0.6
        else:
            length_confidence = 0.8
        
        return (source_confidence + length_confidence) / 2
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get similar documents for a query."""
        return self.retriever.get_relevant_documents(query)
    
    def summarize_documents(self) -> str:
        """Generate a summary of all documents in the vector store."""
        # Get a sample of documents
        sample_docs = self.vector_store.similarity_search("summary overview", k=10)
        
        # Create summary prompt
        doc_contents = "\n\n".join([doc.page_content[:200] for doc in sample_docs])
        
        summary_prompt = f"""
        Provide a comprehensive summary of the following document collection:
        
        {doc_contents}
        
        Include:
        1. Main topics covered
        2. Key insights or findings
        3. Document types and sources
        4. Overall scope and purpose
        """
        
        try:
            summary = self.llm.predict(summary_prompt)
            return summary
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        messages = self.memory.chat_memory.messages
        history = []
        
        for message in messages:
            history.append({
                "type": message.__class__.__name__,
                "content": message.content
            })
        
        return history
