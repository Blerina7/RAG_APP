# Local RAG App - Chat with Your Documents

**Project Bio:** A fully local Retrieval-Augmented Generation (RAG) pipeline built with Python, Ollama, LangChain, and ChromaDB for secure, offline document Q&A.

---

## Project Structure

```text
rag_app/
├── cli.py              # Command-line interface
├── requirements.txt    # Python dependencies
├── src/
│   └── rag_engine.py   # Core RAG logic
├── docs/               # Place your documents here
└── chroma_db/          # Auto-generated vector storage
```

---

## Installation

### 1. Install and Setup Ollama

```bash
# Install Ollama (Linux / macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the default model
ollama pull llama3
```

### 2. Setup Python Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Start Ollama Server

```bash
ollama serve
```

---

## Usage

### Interactive Mode (Recommended)

```bash
python cli.py
```

**In-chat commands:**

* `/ingest docs/file.pdf` - Load a document
* `/stats` - Show database stats
* `/delete filename.pdf` - Remove a document
* `/quit` - Exit

### Direct CLI Commands

```bash
# Load documents
python cli.py ingest docs/report.pdf

# Ask a question
python cli.py query "What does the document say about the requirements?"

# Check stats or delete doc
python cli.py stats
python cli.py delete report.pdf
```

### Switch Models

```bash
python cli.py --model mistral query "Your question"
```

---

## How it Works

```text
Document -> Parse -> Chunk -> Embed -> ChromaDB
                                             |
Query -------------------> Embed -> Search Top-K
                                             |
                       Prompt + Context -> Ollama -> Answer
```

1. **Ingestion:** Documents are parsed, chunked, and stored as vectors in ChromaDB.
2. **Retrieval:** User queries are embedded to find similar chunks via cosine similarity.
3. **Generation:** Relevant chunks + query are sent to Ollama to generate an informed response.

---

## Configuration

Edit the variables inside `cli.py`:

```python
OLLAMA_MODEL = "llama3"
OLLAMA_URL   = "http://localhost:11434"
CHROMA_DIR   = "./chroma_db"
```

---

## Common Issues

* **"Ollama is not running":** Run `ollama serve` in a separate terminal.
* **"Model not found":** Run `ollama pull <model_name>` first.
* **"Missing packages":** Ensure you ran `pip install -r requirements.txt` within your virtual environment.