# 📚 Open-Source RAG with Ollama & Streamlit

A fully open-source, offline-capable Retrieval-Augmented Generation (RAG) application that allows users to upload documents and chat with them using local LLMs.

## 🌟 Features

- **100% Open-Source**: All components use OSI-approved licenses
- **Offline-Capable**: No internet connection required after initial setup
- **Privacy-First**: All data stays on your machine
- **Multiple File Types**: Supports PDF, DOCX, CSV, XLSX
- **Smart Deduplication**: Avoids re-embedding identical files
- **Source Citations**: Shows which documents answers come from
- **Multiple Models**: Choose from llama3, mistral, qwen2.5, phi-3

## 🔒 Open-Source Stack

| Component | Library | License |
|-----------|---------|---------|
| Runtime | Python 3.10+ | PSF |
| LLM Runtime | Ollama | Apache 2.0 |
| Embeddings | nomic-embed-text | Apache 2.0 |
| UI | Streamlit | Apache 2.0 |
| Vector DB | ChromaDB | Apache 2.0 |
| Framework | LangChain | MIT |
| PDF Parser | pypdf | BSD |
| DOCX Parser | python-docx | MIT |

## 📋 Prerequisites

1. **Python 3.10 or higher**
   ```bash
   python --version
   ```

2. **Ollama installed and running**
   
   Install from: https://ollama.ai
   
   ```bash
   # Install Ollama (macOS/Linux)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull required models
   ollama pull llama3
   ollama pull nomic-embed-text
   
   # Optional: Pull additional models
   ollama pull mistral
   ollama pull qwen2.5
   ollama pull phi-3
   ```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MavrickRody/rag_ollama.git
   cd rag_ollama
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload documents**
   - Use the sidebar to upload PDF, DOCX, CSV, or XLSX files
   - Click "Ingest Documents" to process them
   - Wait for confirmation messages

3. **Chat with your documents**
   - Type questions in the chat input
   - Get answers based only on your uploaded documents
   - View source citations for each answer

4. **Change models**
   - Select different models from the sidebar dropdown
   - Supports: llama3, mistral, qwen2.5, phi-3

5. **Clear database**
   - Use "Clear All Documents" to reset the vector database
   - Removes all ingested documents and chat history

## 📁 Project Structure

```
rag_ollama/
│
├── app.py                    # Streamlit UI
├── ingest.py                 # File ingestion & embeddings
├── rag.py                    # Retrieval + QA chain
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── data/
│   ├── uploads/              # User uploaded files
│   └── chroma/               # Persistent vector DB
│
└── utils/
    ├── loaders.py            # OSS document loaders
    └── hashing.py            # Deduplication logic
```

## 🎨 Supported File Types

| Format | Loader | Library |
|--------|--------|---------|
| PDF | PyPDFLoader | pypdf |
| DOCX | Docx2txtLoader | python-docx |
| CSV | CSVLoader | langchain |
| XLSX | UnstructuredExcelLoader | unstructured |

## ⚙️ Configuration

### Chunking Parameters

In `ingest.py`:
```python
chunk_size = 500        # Characters per chunk
chunk_overlap = 100     # Overlap between chunks
```

### Retrieval Parameters

In `rag.py`:
```python
retrieval_k = 4         # Number of chunks to retrieve
```

### Embedding Model

In `ingest.py`:
```python
embedding_model = "nomic-embed-text"  # Ollama embedding model
```

## 🔐 Privacy & Security

- ✅ **100% Offline**: No external API calls
- ✅ **Local Processing**: All data stays on your machine
- ✅ **No Telemetry**: No usage tracking
- ✅ **Open-Source**: Full transparency
- ✅ **No Cloud**: No SaaS dependencies

## 🚫 What's NOT Included

This project deliberately excludes:
- ❌ OpenAI, Anthropic, Google APIs
- ❌ Closed-source models
- ❌ Cloud services
- ❌ Proprietary SDKs
- ❌ Internet dependencies

## 🐛 Troubleshooting

### Ollama not running
```bash
# Check if Ollama is running
curl http://localhost:11434

# Start Ollama (if installed)
ollama serve
```

### Model not found
```bash
# Pull the required model
ollama pull llama3
ollama pull nomic-embed-text
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### ChromaDB errors
```bash
# Clear the database
rm -rf data/chroma
```

## 📝 License

This project uses only OSI-approved open-source licenses:
- Apache 2.0
- MIT
- BSD

## 🤝 Contributing

Contributions must maintain the open-source-only requirement:
- All dependencies must have OSI-approved licenses
- No proprietary APIs or services
- No cloud dependencies
- Must work 100% offline

## 🙏 Acknowledgments

Built with:
- [Ollama](https://ollama.ai) - Local LLM runtime
- [LangChain](https://langchain.com) - RAG framework
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Streamlit](https://streamlit.io) - Web interface

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues first
- Provide error logs and system info
