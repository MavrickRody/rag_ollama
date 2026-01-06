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
    
    # Model selector
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
            st.sidebar.success(f"Model updated to {selected_model}")
    
    st.sidebar.markdown("---")
    
    # File uploader
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
                
                # Initialize RAG engine if we have documents
                if st.session_state.ingestor.has_documents():
                    vectorstore = st.session_state.ingestor.get_vectorstore()
                    st.session_state.rag_engine = RAGEngine(
                        vectorstore,
                        model_name=st.session_state.selected_model
                    )
    
    st.sidebar.markdown("---")
    
    # Clear database button
    st.sidebar.subheader("🗑️ Database")
    if st.sidebar.button("Clear All Documents", type="secondary"):
        st.session_state.ingestor.clear_database()
        st.session_state.rag_engine = None
        st.session_state.messages = []
        st.sidebar.success("Database cleared!")
        # Use st.rerun() for Streamlit >= 1.27, fallback to experimental_rerun
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    
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


def main():
    """Main application entry point."""
    initialize_session_state()
    sidebar()
    main_chat_interface()


if __name__ == "__main__":
    main()
