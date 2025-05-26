"""
Sample code examples for LangChain applications.
"""

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Example 1: Simple LLM Chain
def simple_chain_example():
    """Demonstrate a basic LLM chain."""
    llm = OpenAI(temperature=0.7)
    
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a short paragraph about {topic}."
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run("machine learning")
    return result

# Example 2: Chain with Memory
def memory_chain_example():
    """Demonstrate a chain with conversation memory."""
    llm = OpenAI(temperature=0.7)
    memory = ConversationBufferMemory()
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template="""
        The following is a conversation between a human and an AI assistant.
        
        {chat_history}
        Human: {human_input}
        AI Assistant:"""
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    return chain

# Example 3: Custom Prompt Template
class CustomPromptTemplate(PromptTemplate):
    """Custom prompt template for specialized tasks."""
    
    def format(self, **kwargs):
        # Add custom logic here
        kwargs['context'] = self._get_context(kwargs.get('query', ''))
        return super().format(**kwargs)
    
    def _get_context(self, query):
        # Custom context retrieval logic
        return f"Context for: {query}"

# Example 4: Multi-step Chain
def multi_step_chain():
    """Demonstrate a multi-step reasoning chain."""
    llm = OpenAI(temperature=0.1)
    
    # Step 1: Analyze the problem
    analysis_prompt = PromptTemplate(
        input_variables=["problem"],
        template="Analyze this problem step by step: {problem}"
    )
    
    # Step 2: Generate solution
    solution_prompt = PromptTemplate(
        input_variables=["analysis"],
        template="Based on this analysis: {analysis}\n\nProvide a solution:"
    )
    
    # Create chains
    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
    solution_chain = LLMChain(llm=llm, prompt=solution_prompt)
    
    def solve_problem(problem):
        analysis = analysis_chain.run(problem)
        solution = solution_chain.run(analysis)
        return {
            "problem": problem,
            "analysis": analysis,
            "solution": solution
        }
    
    return solve_problem

# Example 5: Error Handling
def robust_chain_example():
    """Demonstrate error handling in chains."""
    try:
        llm = OpenAI(temperature=0.5)
        
        prompt = PromptTemplate(
            input_variables=["input"],
            template="Process this input: {input}"
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        def safe_run(input_text):
            try:
                result = chain.run(input_text)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return safe_run
        
    except Exception as e:
        print(f"Chain initialization failed: {e}")
        return None

# Example 6: Batch Processing
def batch_processing_example():
    """Process multiple inputs efficiently."""
    llm = OpenAI(temperature=0.3)
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize: {text}"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    def process_batch(texts):
        results = []
        for text in texts:
            try:
                summary = chain.run(text)
                results.append({"input": text[:50] + "...", "summary": summary})
            except Exception as e:
                results.append({"input": text[:50] + "...", "error": str(e)})
        
        return results
    
    return process_batch

# Example 7: Conditional Logic
def conditional_chain_example():
    """Chain with conditional logic based on input."""
    llm = OpenAI(temperature=0.2)
    
    def classify_and_respond(user_input):
        # Simple classification
        if "question" in user_input.lower() or "?" in user_input:
            prompt = PromptTemplate(
                input_variables=["question"],
                template="Answer this question clearly: {question}"
            )
        elif "help" in user_input.lower():
            prompt = PromptTemplate(
                input_variables=["request"],
                template="Provide helpful guidance for: {request}"
            )
        else:
            prompt = PromptTemplate(
                input_variables=["statement"],
                template="Respond appropriately to: {statement}"
            )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(user_input)
    
    return classify_and_respond

# Example 8: Streaming Response
def streaming_example():
    """Demonstrate streaming responses."""
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    
    llm = OpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a detailed explanation about {topic}:"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

if __name__ == "__main__":
    # Test examples
    print("Testing LangChain examples...")
    
    # Note: These examples require valid API keys to run
    # They are provided for demonstration purposes
