# RAG Document Manager

Local RAG (Retrieval-Augmented Generation) system with Web UI and MCP server for OpenClaw integration.

## Features

- **Web UI** — Drag-and-drop upload for PDF and images, document management, semantic search testing
- **MCP Server** — Exposes RAG tools via stdio for Claude/OpenClaw agents
- **PDF Extraction** — Text extraction via pdfplumber (page-level)
- **Image OCR** — pytesseract with Traditional Chinese + English support
- **Vector Database** — ChromaDB with cosine similarity search
- **Local Embeddings** — Ollama `nomic-embed-text` (free, offline, private)

## Architecture

```
server.py         → FastAPI: REST API + Web UI (port 8000)
mcp_server.py     → MCP server: OpenClaw agent tools (stdio)
ingest.py         → Document pipeline: parse → chunk → embed → store
embedder.py       → Ollama embedding wrapper
static/           → Web UI (HTML + vanilla JS)
chroma_db/        → ChromaDB persistent storage (auto-created)
uploads/          → Temporary file storage (auto-created)
```

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- Tesseract OCR (`brew install tesseract`)
- Tesseract Chinese language pack (`brew install tesseract-lang`)

## Setup

```bash
# 1. Pull the embedding model
ollama pull nomic-embed-text

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the web server
python server.py
# Open http://localhost:8000
```

## MCP Server (OpenClaw Integration)

Add to your `openclaw.json` or Claude MCP config:

```json
{
  "mcpServers": {
    "rag": {
      "command": "python",
      "args": ["/path/to/rag/mcp_server.py"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `search(query, n_results)` | Semantic search across all documents |
| `documents()` | List all indexed documents |
| `remove_document(name)` | Delete a document from the knowledge base |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload and ingest a file |
| GET | `/api/documents` | List indexed documents |
| DELETE | `/api/documents/{name}` | Delete a document |
| POST | `/api/search` | Semantic search |

## Supported File Types

- PDF (`.pdf`)
- Images (`.jpg`, `.jpeg`, `.png`, `.webp`)

## Tech Stack

- **ChromaDB** — Local vector database
- **Ollama** — Local embedding model
- **FastAPI** — REST API server
- **pdfplumber** — PDF text extraction
- **pytesseract** — Image OCR
- **MCP** — Model Context Protocol for agent integration
