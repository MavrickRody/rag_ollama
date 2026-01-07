"""
Streamlit UI for Open-Source RAG Application.
100% offline-capable with OSS components only.
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
from ingest import DocumentIngestor
from rag import RAGEngine
from council_rag import AICouncil
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Page configuration
st.set_page_config(
    page_title="Open-Source RAG with Ollama",
    page_icon="📚",
    layout="wide"
)

# Constants
UPLOAD_DIR = "./data/uploads"
CHROMA_DIR = "./data/chroma"
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".csv", ".xlsx", ".xls"]
OSS_MODELS = ["llama3", "mistral", "qwen2.5", "phi-3"]

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "ingestor" not in st.session_state:
        st.session_state.ingestor = DocumentIngestor(
            persist_directory=CHROMA_DIR
        )
    
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = OSS_MODELS[0]
    
    # New session state for modes
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "RAG Mode"  # Options: "RAG Mode", "Chat Only", "AI Council"
    
    if "chat_only_llm" not in st.session_state:
        st.session_state.chat_only_llm = None
    
    # AI Council session state
    if "council_members" not in st.session_state:
        st.session_state.council_members = OSS_MODELS[:2]  # Default: first 2 models
    
    if "discussion_rounds" not in st.session_state:
        st.session_state.discussion_rounds = 1
    
    if "ai_council" not in st.session_state:
        st.session_state.ai_council = None


def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to disk.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Path to saved file
    """
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def sidebar():
    """Render sidebar with controls."""
    st.sidebar.title("📚 Open-Source RAG")
    st.sidebar.markdown("---")
    
    # Mode selector
    st.sidebar.subheader("⚙️ Application Mode")
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["RAG Mode", "Chat Only", "AI Council"],
        index=["RAG Mode", "Chat Only", "AI Council"].index(st.session_state.app_mode)
    )
    
    # Update mode if changed
    if app_mode != st.session_state.app_mode:
        st.session_state.app_mode = app_mode
        st.session_state.messages = []  # Clear messages when switching modes
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Show different controls based on mode
    if app_mode == "AI Council":
        # Council configuration
        st.sidebar.subheader("🏛️ Council Configuration")
        
        selected_members = st.sidebar.multiselect(
            "Select Council Members",
            OSS_MODELS,
            default=st.session_state.council_members
        )
        
        if len(selected_members) < 2:
            st.sidebar.warning("⚠️ Select at least 2 models for council")
        else:
            st.session_state.council_members = selected_members
        
        st.session_state.discussion_rounds = st.sidebar.slider(
            "Discussion Rounds",
            min_value=1,
            max_value=3,
            value=st.session_state.discussion_rounds,
            help="Number of discussion rounds for council members"
        )
        
        st.sidebar.markdown("---")
        
        # Initialize council if members changed
        if st.session_state.app_mode == "AI Council":
            has_docs = st.session_state.ingestor.has_documents()
            vectorstore = st.session_state.ingestor.get_vectorstore() if has_docs else None
            
            # Only create council if we have at least 2 members
            if len(st.session_state.council_members) >= 2:
                st.session_state.ai_council = AICouncil(
                    model_names=st.session_state.council_members,
                    vectorstore=vectorstore
                )
    
    else:
        # Model selector (for RAG Mode and Chat Only)
        st.sidebar.subheader("🧠 LLM Model")
        selected_model = st.sidebar.selectbox(
            "Select Model (OSS only)",
            OSS_MODELS,
            index=OSS_MODELS.index(st.session_state.selected_model)
        )
        
        # Update model if changed
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            if st.session_state.rag_engine is not None:
                st.session_state.rag_engine.update_model(selected_model)
            if st.session_state.chat_only_llm is not None:
                st.session_state.chat_only_llm = Ollama(model=selected_model)
            st.sidebar.success(f"Model updated to {selected_model}")
        
        # Initialize chat-only LLM if in Chat Only mode
        if app_mode == "Chat Only" and st.session_state.chat_only_llm is None:
            st.session_state.chat_only_llm = Ollama(model=st.session_state.selected_model)
        
        st.sidebar.markdown("---")
    
    # File uploader (shown for RAG Mode and AI Council)
    if app_mode in ["RAG Mode", "AI Council"]:
        st.sidebar.subheader("📁 Upload Documents")
        uploaded_files = st.sidebar.file_uploader(
            "Choose files",
            type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True
        )
        
        # Ingest button
        if uploaded_files:
            if st.sidebar.button("🚀 Ingest Documents", type="primary"):
                with st.sidebar:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    for idx, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing: {uploaded_file.name}")
                        
                        # Save file
                        file_path = save_uploaded_file(uploaded_file)
                        
                        # Ingest file
                        result = st.session_state.ingestor.ingest_file(
                            file_path, uploaded_file.name
                        )
                        results.append(result)
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Show results
                    for result in results:
                        if result["status"] == "success":
                            st.success(f"✅ {result['filename']}: {result['message']}")
                        elif result["status"] == "skipped":
                            st.info(f"⏭️ {result['filename']}: {result['message']}")
                        else:
                            st.error(f"❌ {result['filename']}: {result['message']}")
                    
                    # Initialize RAG engine if we have documents (for RAG mode)
                    if st.session_state.ingestor.has_documents():
                        vectorstore = st.session_state.ingestor.get_vectorstore()
                        if app_mode == "RAG Mode":
                            st.session_state.rag_engine = RAGEngine(
                                vectorstore,
                                model_name=st.session_state.selected_model
                            )
                        elif app_mode == "AI Council" and len(st.session_state.council_members) >= 2:
                            st.session_state.ai_council = AICouncil(
                                model_names=st.session_state.council_members,
                                vectorstore=vectorstore
                            )
        
        st.sidebar.markdown("---")
        
        # Clear database button
        st.sidebar.subheader("🗑️ Database")
        if st.sidebar.button("Clear All Documents", type="secondary"):
            st.session_state.ingestor.clear_database()
            st.session_state.rag_engine = None
            st.session_state.ai_council = None
            st.session_state.messages = []
            st.sidebar.success("Database cleared!")
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Status info
        st.sidebar.subheader("ℹ️ Status")
        has_docs = st.session_state.ingestor.has_documents()
        if has_docs:
            st.sidebar.success("✅ Documents loaded")
            num_files = len(st.session_state.ingestor.ingested_hashes)
            st.sidebar.info(f"📊 {num_files} file(s) ingested")
        else:
            st.sidebar.warning("⚠️ No documents loaded")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("🔒 100% Offline • 🌟 Open-Source Only")


def main_chat_interface():
    """Render main chat interface."""
    st.title("💬 Chat with Your Documents")
    
    # Check if documents are loaded
    if not st.session_state.ingestor.has_documents():
        st.info("👈 Please upload and ingest documents using the sidebar to start chatting.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📎 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Query RAG engine
                result = st.session_state.rag_engine.query(prompt)
                
                if result["success"]:
                    st.markdown(result["answer"])
                    
                    # Show sources
                    if result["sources"]:
                        with st.expander("📎 Sources"):
                            for source in result["sources"]:
                                st.markdown(f"- {source}")
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                else:
                    error_msg = result["answer"]
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })


def chat_only_interface():
    """Render chat-only interface (no file upload required)."""
    st.title("💬 Chat with AI Model")
    
    st.info(f"🤖 Chatting with **{st.session_state.selected_model}** in direct mode (no documents)")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create simple prompt template
                    template = """You are a helpful AI assistant.

Question: {question}

Answer:"""
                    prompt_template = PromptTemplate(
                        template=template,
                        input_variables=["question"]
                    )
                    
                    # Create chain
                    chain = prompt_template | st.session_state.chat_only_llm | StrOutputParser()
                    
                    # Get response
                    response = chain.invoke({"question": prompt})
                    
                    st.markdown(response)
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def ai_council_interface():
    """Render AI Council interface."""
    st.title("🏛️ AI Council Discussion")
    
    # Check if council is configured
    if len(st.session_state.council_members) < 2:
        st.warning("👈 Please select at least 2 models in the sidebar to start a council discussion.")
        return
    
    # Show council info
    has_docs = st.session_state.ingestor.has_documents()
    if has_docs:
        st.info(f"🤖 Council with **{len(st.session_state.council_members)} members** ({', '.join(st.session_state.council_members)}) • Using RAG with {len(st.session_state.ingestor.ingested_hashes)} document(s)")
    else:
        st.info(f"🤖 Council with **{len(st.session_state.council_members)} members** ({', '.join(st.session_state.council_members)}) • Direct chat mode")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # Display council discussion
                if "discussion_history" in message:
                    # Show discussion rounds
                    for round_data in message["discussion_history"]:
                        st.markdown(f"### 🔄 Round {round_data['round']}")
                        
                        for model_name, response in round_data["responses"].items():
                            with st.expander(f"**{model_name}**", expanded=(round_data['round'] == 1)):
                                st.markdown(response)
                    
                    # Show consensus
                    st.markdown("### 🎯 Consensus Summary")
                    st.markdown(message["consensus"])
                    
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander("📎 Sources"):
                            for source in message["sources"]:
                                st.markdown(f"- {source}")
                else:
                    st.markdown(message.get("content", ""))
    
    # Chat input
    if prompt := st.chat_input("Ask the council a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate council response
        with st.chat_message("assistant"):
            with st.spinner(f"Council discussing ({st.session_state.discussion_rounds} round(s))..."):
                # Query AI Council
                result = st.session_state.ai_council.discuss(
                    prompt,
                    discussion_rounds=st.session_state.discussion_rounds
                )
                
                if result["success"]:
                    # Display discussion rounds
                    for round_data in result["discussion_history"]:
                        st.markdown(f"### 🔄 Round {round_data['round']}")
                        
                        for model_name, response in round_data["responses"].items():
                            with st.expander(f"**{model_name}**", expanded=(round_data['round'] == 1)):
                                st.markdown(response)
                    
                    # Display consensus
                    st.markdown("### 🎯 Consensus Summary")
                    st.markdown(result["consensus"])
                    
                    # Show sources
                    if result["sources"]:
                        with st.expander("📎 Sources"):
                            for source in result["sources"]:
                                st.markdown(f"- {source}")
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "discussion_history": result["discussion_history"],
                        "consensus": result["consensus"],
                        "sources": result["sources"]
                    })
                else:
                    error_msg = result.get("consensus", "Error during council discussion")
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })



def main():
    """Main application entry point."""
    initialize_session_state()
    sidebar()
    
    # Route to appropriate interface based on mode
    if st.session_state.app_mode == "RAG Mode":
        main_chat_interface()
    elif st.session_state.app_mode == "Chat Only":
        chat_only_interface()
    elif st.session_state.app_mode == "AI Council":
        ai_council_interface()


if __name__ == "__main__":
    main()
