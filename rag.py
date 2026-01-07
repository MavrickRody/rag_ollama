"""
RAG (Retrieval-Augmented Generation) module.
Handles retrieval and question-answering chain.
"""
from typing import Optional, Dict, List
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Hallucination-safe prompt template
RAG_PROMPT_TEMPLATE = """You are an open-source AI assistant.

Answer the question using ONLY the provided context.
If the answer cannot be found, say exactly:
"I don't know based on the uploaded documents."

Context:
{context}

Question:
{question}

Answer:"""


class RAGEngine:
    """Handles retrieval and question-answering."""
    
    def __init__(
        self,
        vectorstore: Chroma,
        model_name: str = "llama3",
        retrieval_k: int = 4
    ):
        """
        Initialize the RAG engine.
        
        Args:
            vectorstore: ChromaDB vector store instance
            model_name: Ollama model to use for generation
            retrieval_k: Number of documents to retrieve
        """
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.retrieval_k = retrieval_k
        
        # Initialize LLM (OSS - Ollama)
        self.llm = Ollama(model=model_name)
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retrieval_k}
        )
        
        # Create RAG chain using LCEL (LangChain Expression Language)
        self.qa_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            # Get the answer
            answer = self.qa_chain.invoke(question)
            
            # Retrieve source documents separately for citations
            source_docs = self.retriever.invoke(question)
            
            # Extract sources
            sources = []
            seen_sources = set()
            for doc in source_docs:
                source_file = doc.metadata.get("source_file", "Unknown")
                page = doc.metadata.get("page", None)
                
                # Create source string
                if page is not None:
                    source_str = f"{source_file} (page {page + 1})"
                else:
                    source_str = source_file
                
                if source_str not in seen_sources:
                    sources.append(source_str)
                    seen_sources.add(source_str)
            
            return {
                "answer": answer if answer else "No answer generated.",
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def update_model(self, model_name: str):
        """
        Update the LLM model.
        
        Args:
            model_name: New Ollama model name
        """
        self.model_name = model_name
        self.llm = Ollama(model=model_name)
        
        # Recreate the chain with new model
        self.qa_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
