# RAG MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local RAG system with MCP server for OpenClaw and a Web UI for document ingestion.

**Architecture:** FastAPI serves both the Web UI and REST API. A separate MCP server process exposes search/list/delete tools for OpenClaw agents. ChromaDB stores vectors locally, Ollama provides embeddings, pdfplumber handles PDFs, pytesseract handles image OCR.

**Tech Stack:** Python 3.9+, FastAPI, ChromaDB, Ollama (nomic-embed-text), pdfplumber, pytesseract, MCP Python SDK

**Spec:** `docs/superpowers/specs/2026-03-25-rag-mcp-server-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignore chroma_db/, uploads/, __pycache__/, .venv/ |
| `embedder.py` | Ollama embedding wrapper — single function `get_embeddings(texts)` |
| `ingest.py` | Document parsing (PDF/image) + chunking + embedding + ChromaDB storage |
| `server.py` | FastAPI app: REST API endpoints + static file serving |
| `mcp_server.py` | MCP server with search/list/delete tools |
| `static/index.html` | Web UI — drag-drop upload, document list, search test |
| `static/app.js` | Frontend JS — API calls, DOM updates |

---

## Task 0: Prerequisites & Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Install system dependencies**

```bash
brew install ollama tesseract
```

- [ ] **Step 2: Start Ollama and pull embedding model**

```bash
ollama serve &  # if not already running
ollama pull nomic-embed-text
```

Verify: `ollama list` shows `nomic-embed-text`

- [ ] **Step 3: Create requirements.txt**

```
chromadb>=0.5.0
fastapi>=0.115.0
uvicorn>=0.34.0
python-multipart>=0.0.20
pdfplumber>=0.11.0
pytesseract>=0.3.13
Pillow>=11.0.0
mcp>=1.0.0
httpx>=0.28.0
```

- [ ] **Step 4: Create .gitignore**

```
chroma_db/
uploads/
__pycache__/
*.pyc
.venv/
.env
```

- [ ] **Step 5: Create virtualenv and install dependencies**

```bash
cd /Users/mike/Desktop/rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 6: Create uploads directory**

```bash
mkdir -p uploads
```

- [ ] **Step 7: Init git repo and commit**

```bash
cd /Users/mike/Desktop/rag
git init
git add requirements.txt .gitignore docs/
git commit -m "chore: init project with requirements and spec"
```

---

## Task 1: Embedding Wrapper

**Files:**
- Create: `embedder.py`

- [ ] **Step 1: Write embedder.py**

```python
"""Ollama embedding wrapper for nomic-embed-text."""

import httpx

OLLAMA_BASE = "http://localhost:11434"
MODEL = "nomic-embed-text"


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts via Ollama API."""
    embeddings = []
    for text in texts:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": MODEL, "input": text},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings.append(data["embeddings"][0])
    return embeddings


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    return get_embeddings([text])[0]
```

- [ ] **Step 2: Verify embedder works**

```bash
source .venv/bin/activate
python3 -c "
from embedder import get_embedding
vec = get_embedding('hello world')
print(f'Dimension: {len(vec)}')
assert len(vec) == 768, f'Expected 768, got {len(vec)}'
print('OK')
"
```

Expected: `Dimension: 768` then `OK`

- [ ] **Step 3: Commit**

```bash
git add embedder.py
git commit -m "feat: add Ollama embedding wrapper"
```

---

## Task 2: Document Ingestion Pipeline

**Files:**
- Create: `ingest.py`

- [ ] **Step 1: Write ingest.py**

```python
"""Document ingestion: parse → chunk → embed → store in ChromaDB."""

import os
import uuid
from datetime import datetime, timezone

import chromadb
import pdfplumber
import pytesseract
from PIL import Image

from embedder import get_embeddings

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """Extract text from PDF, one entry per page."""
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": i + 1})
    return pages


def extract_text_from_image(file_path: str) -> list[dict]:
    """OCR an image file."""
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img, lang="chi_tra+eng")
    if text.strip():
        return [{"text": text, "page": 1}]
    return []


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def ingest_file(file_path: str) -> dict:
    """Ingest a file into ChromaDB. Returns metadata about the ingestion."""
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    # Extract text
    if ext == ".pdf":
        pages = extract_text_from_pdf(file_path)
        doc_type = "pdf"
    elif ext in (".jpg", ".jpeg", ".png", ".webp"):
        pages = extract_text_from_image(file_path)
        doc_type = "image"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not pages:
        raise ValueError(f"No text extracted from {filename}")

    # Chunk all pages
    all_chunks = []
    for page_data in pages:
        chunks = chunk_text(page_data["text"])
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "page": page_data["page"],
            })

    if not all_chunks:
        raise ValueError(f"No chunks generated from {filename}")

    # Embed
    texts = [c["text"] for c in all_chunks]
    embeddings = get_embeddings(texts)

    # Store in ChromaDB
    collection = get_collection()
    now = datetime.now(timezone.utc).isoformat()
    ids = [f"{filename}_{uuid.uuid4().hex[:8]}" for _ in all_chunks]
    metadatas = [
        {
            "source": filename,
            "page": c["page"],
            "type": doc_type,
            "added_at": now,
        }
        for c in all_chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return {
        "name": filename,
        "type": doc_type,
        "chunks": len(all_chunks),
        "added_at": now,
    }


def search_documents(query: str, n_results: int = 5) -> list[dict]:
    """Search documents by semantic similarity."""
    from embedder import get_embedding

    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "score": round(1 - results["distances"][0][i], 4),
        })
    return output


def list_documents() -> list[dict]:
    """List all unique documents in the collection."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    all_data = collection.get(include=["metadatas"])
    docs = {}
    for meta in all_data["metadatas"]:
        name = meta["source"]
        if name not in docs:
            docs[name] = {
                "name": name,
                "type": meta["type"],
                "chunks": 0,
                "added_at": meta["added_at"],
            }
        docs[name]["chunks"] += 1

    return sorted(docs.values(), key=lambda d: d["added_at"], reverse=True)


def delete_document(document_name: str) -> bool:
    """Delete all chunks for a document. Returns True if found and deleted."""
    collection = get_collection()
    all_data = collection.get(include=["metadatas"])

    ids_to_delete = [
        id_ for id_, meta in zip(all_data["ids"], all_data["metadatas"])
        if meta["source"] == document_name
    ]

    if not ids_to_delete:
        return False

    collection.delete(ids=ids_to_delete)
    return True
```

- [ ] **Step 2: Verify ingestion with a test PDF**

```bash
source .venv/bin/activate
python3 -c "
from ingest import ingest_file, search_documents, list_documents
result = ingest_file('/Users/mike/Desktop/pdf/backpack.pdf')
print(f'Ingested: {result}')
docs = list_documents()
print(f'Documents: {docs}')
results = search_documents('backpack')
print(f'Search results: {len(results)}')
for r in results[:2]:
    print(f'  [{r[\"score\"]}] {r[\"source\"]} p{r[\"page\"]}: {r[\"content\"][:80]}...')
"
```

Expected: Shows ingested document info, document list, and search results.

- [ ] **Step 3: Clean up test data and commit**

```bash
rm -rf chroma_db/
git add ingest.py
git commit -m "feat: add document ingestion pipeline (PDF + OCR + ChromaDB)"
```

---

## Task 3: FastAPI Server + REST API

**Files:**
- Create: `server.py`

- [ ] **Step 1: Write server.py**

```python
"""FastAPI server: REST API + static Web UI."""

import os
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingest import ingest_file, search_documents, list_documents, delete_document

app = FastAPI(title="RAG Document Manager")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_file(file_path)
        return result
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, str(e))


@app.get("/api/documents")
async def get_documents():
    return list_documents()


@app.delete("/api/documents/{name:path}")
async def remove_document(name: str):
    deleted = delete_document(name)
    if not deleted:
        raise HTTPException(404, f"Document not found: {name}")
    # Also remove uploaded file if exists
    file_path = os.path.join(UPLOAD_DIR, name)
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"deleted": name}


@app.post("/api/search")
async def search(req: SearchRequest):
    return search_documents(req.query, req.n_results)


# Serve static files (Web UI)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 2: Quick smoke test**

```bash
source .venv/bin/activate
# Create placeholder static files so server starts
mkdir -p static
echo "<html><body>placeholder</body></html>" > static/index.html
python3 -c "
import uvicorn
from server import app
# Just verify it imports and creates the app
print(f'App routes: {[r.path for r in app.routes]}')
print('OK')
"
rm static/index.html
```

Expected: Lists routes including `/api/upload`, `/api/documents`, `/api/search`, `/`

- [ ] **Step 3: Commit**

```bash
git add server.py
git commit -m "feat: add FastAPI server with REST API endpoints"
```

---

## Task 4: MCP Server

**Files:**
- Create: `mcp_server.py`

- [ ] **Step 1: Write mcp_server.py**

```python
"""MCP server for OpenClaw — exposes RAG tools via stdio."""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP

from ingest import search_documents, list_documents, delete_document

mcp = FastMCP("rag")


@mcp.tool()
def search(query: str, n_results: int = 5) -> str:
    """Search the RAG knowledge base by semantic similarity.
    Returns the most relevant document chunks matching the query.
    Use this to find information from uploaded PDFs and images."""
    import json
    results = search_documents(query, n_results)
    if not results:
        return "No documents in the knowledge base yet."
    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
def documents() -> str:
    """List all documents indexed in the RAG knowledge base.
    Shows document name, type, number of chunks, and when it was added."""
    import json
    docs = list_documents()
    if not docs:
        return "No documents indexed yet."
    return json.dumps(docs, ensure_ascii=False, indent=2)


@mcp.tool()
def remove_document(document_name: str) -> str:
    """Remove a document and all its chunks from the RAG knowledge base."""
    deleted = delete_document(document_name)
    if deleted:
        return f"Deleted '{document_name}' from knowledge base."
    return f"Document '{document_name}' not found."


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- [ ] **Step 2: Verify MCP server starts**

```bash
source .venv/bin/activate
python3 -c "
from mcp_server import mcp
print(f'Server name: {mcp.name}')
print('OK - MCP server configured')
"
```

Expected: `Server name: rag` then `OK`

- [ ] **Step 3: Commit**

```bash
git add mcp_server.py
git commit -m "feat: add MCP server with search/list/delete tools"
```

---

## Task 5: Web UI

**Files:**
- Create: `static/index.html`
- Create: `static/app.js`

- [ ] **Step 1: Write static/index.html**

```html
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Document Manager</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #1a1a2e; color: #e0e0e0;
            min-height: 100vh; padding: 2rem;
        }
        h1 { color: #e94560; margin-bottom: 1.5rem; font-size: 1.8rem; }
        h2 { color: #e94560; margin: 1.5rem 0 0.8rem; font-size: 1.2rem; }

        /* Upload zone */
        .upload-zone {
            border: 2px dashed #e94560; border-radius: 12px;
            padding: 3rem; text-align: center; cursor: pointer;
            transition: all 0.3s; background: #16213e;
        }
        .upload-zone:hover, .upload-zone.dragover {
            background: #1a1a40; border-color: #ff6b6b;
        }
        .upload-zone p { font-size: 1.1rem; }
        .upload-zone .hint { color: #888; font-size: 0.85rem; margin-top: 0.5rem; }
        .upload-zone input { display: none; }

        /* Status message */
        .status {
            padding: 0.8rem; border-radius: 8px; margin: 1rem 0;
            display: none; font-size: 0.9rem;
        }
        .status.success { display: block; background: #0f3d0f; color: #4caf50; }
        .status.error { display: block; background: #3d0f0f; color: #f44336; }
        .status.loading { display: block; background: #1a1a40; color: #ffc107; }

        /* Document list */
        .doc-list { list-style: none; }
        .doc-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.8rem 1rem; margin: 0.5rem 0;
            background: #16213e; border-radius: 8px;
        }
        .doc-item .info { flex: 1; }
        .doc-item .name { font-weight: 600; }
        .doc-item .meta { color: #888; font-size: 0.8rem; }
        .doc-item button {
            background: #e94560; color: white; border: none;
            padding: 0.4rem 0.8rem; border-radius: 6px; cursor: pointer;
        }
        .doc-item button:hover { background: #ff6b6b; }

        /* Search */
        .search-box {
            display: flex; gap: 0.5rem; margin: 0.8rem 0;
        }
        .search-box input {
            flex: 1; padding: 0.6rem 1rem; border: 1px solid #333;
            border-radius: 8px; background: #16213e; color: #e0e0e0;
            font-size: 1rem;
        }
        .search-box button {
            padding: 0.6rem 1.2rem; background: #e94560; color: white;
            border: none; border-radius: 8px; cursor: pointer; font-size: 1rem;
        }
        .search-box button:hover { background: #ff6b6b; }

        .result-item {
            background: #16213e; border-radius: 8px;
            padding: 1rem; margin: 0.5rem 0;
        }
        .result-item .source {
            color: #e94560; font-size: 0.8rem; font-weight: 600;
        }
        .result-item .content {
            margin-top: 0.4rem; line-height: 1.5; white-space: pre-wrap;
        }
        .result-item .score { color: #888; font-size: 0.8rem; }

        .empty { color: #666; font-style: italic; padding: 1rem 0; }

        @media (max-width: 600px) {
            body { padding: 1rem; }
            .upload-zone { padding: 2rem 1rem; }
        }
    </style>
</head>
<body>
    <h1>📚 RAG Document Manager</h1>

    <div class="upload-zone" id="uploadZone">
        <p>拖放 PDF / 圖片 至此，或點擊選擇檔案</p>
        <p class="hint">支援: PDF, JPG, PNG, WEBP</p>
        <input type="file" id="fileInput" multiple accept=".pdf,.jpg,.jpeg,.png,.webp">
    </div>

    <div class="status" id="status"></div>

    <h2>已索引文件</h2>
    <ul class="doc-list" id="docList">
        <li class="empty">載入中...</li>
    </ul>

    <h2>搜索測試</h2>
    <div class="search-box">
        <input type="text" id="searchInput" placeholder="輸入搜索內容..." />
        <button id="searchBtn">搜索</button>
    </div>
    <div id="searchResults"></div>

    <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write static/app.js**

```javascript
const API = '';

// Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const status = document.getElementById('status');
const docList = document.getElementById('docList');
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchResults = document.getElementById('searchResults');

// Status helpers
function showStatus(msg, type) {
    status.textContent = msg;
    status.className = 'status ' + type;
}
function hideStatus() { status.className = 'status'; }

// Upload
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

async function handleFiles(files) {
    for (const file of files) {
        showStatus(`上傳中: ${file.name}...`, 'loading');
        const form = new FormData();
        form.append('file', file);
        try {
            const res = await fetch(`${API}/api/upload`, { method: 'POST', body: form });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || res.statusText);
            }
            const data = await res.json();
            showStatus(`✓ ${file.name} — ${data.chunks} 個片段已索引`, 'success');
        } catch (e) {
            showStatus(`✗ ${file.name}: ${e.message}`, 'error');
        }
    }
    loadDocuments();
    fileInput.value = '';
}

// Documents
async function loadDocuments() {
    try {
        const res = await fetch(`${API}/api/documents`);
        const docs = await res.json();
        if (docs.length === 0) {
            docList.innerHTML = '<li class="empty">尚無文件，請上傳 PDF 或圖片</li>';
            return;
        }
        docList.innerHTML = docs.map(d => `
            <li class="doc-item">
                <div class="info">
                    <span class="name">${d.type === 'pdf' ? '📄' : '🖼'} ${d.name}</span>
                    <span class="meta">${d.chunks} chunks · ${new Date(d.added_at).toLocaleString('zh-TW')}</span>
                </div>
                <button onclick="deleteDoc('${d.name}')">刪除</button>
            </li>
        `).join('');
    } catch (e) {
        docList.innerHTML = `<li class="empty">載入失敗: ${e.message}</li>`;
    }
}

async function deleteDoc(name) {
    if (!confirm(`確定刪除 ${name}？`)) return;
    showStatus(`刪除中: ${name}...`, 'loading');
    try {
        const res = await fetch(`${API}/api/documents/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error((await res.json()).detail);
        showStatus(`✓ 已刪除 ${name}`, 'success');
    } catch (e) {
        showStatus(`✗ ${e.message}`, 'error');
    }
    loadDocuments();
}

// Search
searchBtn.addEventListener('click', doSearch);
searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

async function doSearch() {
    const query = searchInput.value.trim();
    if (!query) return;
    searchResults.innerHTML = '<div class="empty">搜索中...</div>';
    try {
        const res = await fetch(`${API}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, n_results: 5 }),
        });
        const results = await res.json();
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="empty">無搜索結果</div>';
            return;
        }
        searchResults.innerHTML = results.map(r => `
            <div class="result-item">
                <span class="source">${r.source} · 第${r.page}頁</span>
                <span class="score">相似度: ${(r.score * 100).toFixed(1)}%</span>
                <div class="content">${escapeHtml(r.content)}</div>
            </div>
        `).join('');
    } catch (e) {
        searchResults.innerHTML = `<div class="empty">搜索失敗: ${e.message}</div>`;
    }
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

// Init
loadDocuments();
```

- [ ] **Step 3: Commit**

```bash
git add static/
git commit -m "feat: add Web UI with drag-drop upload and search"
```

---

## Task 6: Integration Test & GitHub Push

- [ ] **Step 1: Start the server and test full workflow**

```bash
source .venv/bin/activate
python3 server.py &
SERVER_PID=$!
sleep 2

# Test upload
curl -s -F "file=@/Users/mike/Desktop/pdf/backpack.pdf" http://localhost:8000/api/upload | python3 -m json.tool

# Test list
curl -s http://localhost:8000/api/documents | python3 -m json.tool

# Test search
curl -s -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "backpack", "n_results": 3}' | python3 -m json.tool

# Cleanup
kill $SERVER_PID
rm -rf chroma_db/ uploads/*
```

Expected: All three API calls return valid JSON responses.

- [ ] **Step 2: Test MCP server connection**

```bash
source .venv/bin/activate
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test"}},"id":1}' | python3 mcp_server.py
```

Expected: JSON response with server capabilities.

- [ ] **Step 3: Create GitHub repo and push**

```bash
cd /Users/mike/Desktop/rag
gh repo create rag --public --source=. --push
```

- [ ] **Step 4: Update openclaw.json with MCP server config**

Add to the relevant agent's MCP servers section:
```json
{
  "mcpServers": {
    "rag": {
      "command": "/Users/mike/Desktop/rag/.venv/bin/python3",
      "args": ["/Users/mike/Desktop/rag/mcp_server.py"]
    }
  }
}
```

Note: Use the venv python to ensure all dependencies are available.

---

## Summary

| Task | Description | Key Files |
|------|------------|-----------|
| 0 | Prerequisites & project setup | requirements.txt, .gitignore |
| 1 | Embedding wrapper | embedder.py |
| 2 | Document ingestion pipeline | ingest.py |
| 3 | FastAPI server + REST API | server.py |
| 4 | MCP server for OpenClaw | mcp_server.py |
| 5 | Web UI | static/index.html, static/app.js |
| 6 | Integration test & GitHub push | — |
