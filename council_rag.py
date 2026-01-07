"""
AI Council module for multi-model discussion and consensus.
Enables multiple models to discuss queries and reach consensus.
"""
from typing import List, Dict, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser


# Prompt templates for council
COUNCIL_RAG_PROMPT = """You are an AI assistant participating in a council discussion.

Answer the question using ONLY the provided context.
If the answer cannot be found, say exactly:
"I don't know based on the uploaded documents."

Context:
{context}

Question:
{question}

Answer:"""


COUNCIL_CHAT_PROMPT = """You are an AI assistant participating in a council discussion.

Question:
{question}

Answer:"""


COUNCIL_DISCUSSION_PROMPT = """You are an AI assistant participating in a council discussion.

Original Question:
{question}

Other Council Members' Responses:
{other_responses}

Based on the other council members' responses, provide your refined perspective.
Consider areas of agreement and disagreement.

Your Response:"""


CONSENSUS_PROMPT = """You are tasked with creating a consensus summary from multiple AI responses.

Question:
{question}

Council Members' Responses:
{all_responses}

Create a consensus summary that:
1. Highlights points of agreement among council members
2. Notes any significant differences or alternative perspectives
3. Provides a balanced recommendation or conclusion

Consensus Summary:"""


class AICouncil:
    """Handles multi-model discussion and consensus generation."""
    
    def __init__(
        self,
        model_names: List[str],
        vectorstore: Optional[Chroma] = None,
        retrieval_k: int = 4
    ):
        """
        Initialize the AI Council.
        
        Args:
            model_names: List of Ollama model names for council members
            vectorstore: Optional ChromaDB vector store for RAG mode
            retrieval_k: Number of documents to retrieve in RAG mode
        """
        self.model_names = model_names
        self.vectorstore = vectorstore
        self.retrieval_k = retrieval_k
        self.use_rag = vectorstore is not None
        
        # Initialize LLMs for each council member
        self.llms = {
            model_name: Ollama(model=model_name)
            for model_name in model_names
        }
        
        # Create retriever if using RAG
        self.retriever = None
        if self.use_rag:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.retrieval_k}
            )
    
    def _format_docs(self, docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _get_initial_response(self, model_name: str, question: str, context: str = None) -> str:
        """
        Get initial response from a model.
        
        Args:
            model_name: Name of the model
            question: User question
            context: Retrieved context (for RAG mode)
            
        Returns:
            Model's response
        """
        llm = self.llms[model_name]
        
        if self.use_rag and context:
            prompt = PromptTemplate(
                template=COUNCIL_RAG_PROMPT,
                input_variables=["context", "question"]
            )
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"context": context, "question": question})
        else:
            prompt = PromptTemplate(
                template=COUNCIL_CHAT_PROMPT,
                input_variables=["question"]
            )
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"question": question})
    
    def _get_discussion_response(
        self,
        model_name: str,
        question: str,
        other_responses: str
    ) -> str:
        """
        Get a model's response after seeing other models' responses.
        
        Args:
            model_name: Name of the model
            question: Original question
            other_responses: Formatted responses from other models
            
        Returns:
            Model's refined response
        """
        llm = self.llms[model_name]
        prompt = PromptTemplate(
            template=COUNCIL_DISCUSSION_PROMPT,
            input_variables=["question", "other_responses"]
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "question": question,
            "other_responses": other_responses
        })
    
    def _generate_consensus(self, question: str, all_responses: str) -> str:
        """
        Generate consensus summary from all responses.
        
        Args:
            question: Original question
            all_responses: All council members' responses
            
        Returns:
            Consensus summary
        """
        # Use the first model to generate consensus
        llm = self.llms[self.model_names[0]]
        prompt = PromptTemplate(
            template=CONSENSUS_PROMPT,
            input_variables=["question", "all_responses"]
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "question": question,
            "all_responses": all_responses
        })
    
    def discuss(
        self,
        question: str,
        discussion_rounds: int = 1
    ) -> Dict:
        """
        Conduct council discussion on a question.
        
        Args:
            question: User question
            discussion_rounds: Number of discussion rounds
            
        Returns:
            Dictionary with discussion results
        """
        try:
            # Retrieve context if using RAG
            context = None
            sources = []
            if self.use_rag and self.retriever:
                source_docs = self.retriever.invoke(question)
                context = self._format_docs(source_docs)
                
                # Extract sources
                seen_sources = set()
                for doc in source_docs:
                    source_file = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", None)
                    
                    if page is not None:
                        source_str = f"{source_file} (page {page + 1})"
                    else:
                        source_str = source_file
                    
                    if source_str not in seen_sources:
                        sources.append(source_str)
                        seen_sources.add(source_str)
            
            # Store all responses across rounds
            discussion_history = []
            
            # Round 1: Initial responses
            round_responses = {}
            for model_name in self.model_names:
                response = self._get_initial_response(model_name, question, context)
                round_responses[model_name] = response
            
            discussion_history.append({
                "round": 1,
                "responses": round_responses.copy()
            })
            
            # Additional discussion rounds
            for round_num in range(2, discussion_rounds + 1):
                # Format other responses for each model
                new_round_responses = {}
                
                for model_name in self.model_names:
                    # Get responses from other models
                    other_models_responses = "\n\n".join([
                        f"**{other_model}**: {response}"
                        for other_model, response in round_responses.items()
                        if other_model != model_name
                    ])
                    
                    # Get refined response
                    refined_response = self._get_discussion_response(
                        model_name,
                        question,
                        other_models_responses
                    )
                    new_round_responses[model_name] = refined_response
                
                round_responses = new_round_responses
                discussion_history.append({
                    "round": round_num,
                    "responses": round_responses.copy()
                })
            
            # Generate consensus
            all_responses_formatted = "\n\n".join([
                f"**{model_name}**: {response}"
                for model_name, response in round_responses.items()
            ])
            
            consensus = self._generate_consensus(question, all_responses_formatted)
            
            return {
                "success": True,
                "discussion_history": discussion_history,
                "consensus": consensus,
                "sources": sources,
                "num_rounds": discussion_rounds,
                "num_members": len(self.model_names)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "discussion_history": [],
                "consensus": f"Error during council discussion: {str(e)}",
                "sources": []
            }
