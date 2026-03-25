# RAG MCP Server for OpenClaw — Design Spec

## Overview

本地 RAG（Retrieval-Augmented Generation）系統，提供：
1. **MCP Server** — openclaw agent 透過 MCP 工具查詢知識庫
2. **Web UI** — 瀏覽器拖放上傳 PDF/圖片，管理已索引文件
3. **文件處理管道** — PDF 文字擷取 + 圖片 OCR → 切塊 → Ollama embedding → ChromaDB

## Tech Stack

| 組件 | 技術 | 說明 |
|------|------|------|
| Vector DB | ChromaDB | 本地檔案儲存，無需 Docker |
| Embedding | Ollama `nomic-embed-text` | 本地免費，768 維 |
| PDF 解析 | pdfplumber | 擷取文字和表格 |
| 圖片 OCR | pytesseract + Pillow | 圖片轉文字 |
| Web 框架 | FastAPI + uvicorn | REST API + 靜態檔案服務 |
| MCP | mcp Python SDK | stdio 傳輸，供 openclaw 連接 |
| 前端 | 原生 HTML/CSS/JS | 無框架，無 build step |

## Architecture

```
/Users/mike/Desktop/rag/
├── server.py           # FastAPI app (Web UI + REST API)
├── mcp_server.py       # MCP server (openclaw 連接點)
├── ingest.py           # 文件解析 + 切塊 + embedding + 儲存
├── embedder.py         # Ollama embedding 封裝
├── static/
│   ├── index.html      # 拖放上傳介面
│   └── app.js          # 前端邏輯
├── uploads/            # 暫存上傳檔案
├── chroma_db/          # ChromaDB 向量資料 (gitignored)
├── requirements.txt
└── .gitignore
```

## MCP Server Tools

### `search_documents(query: str, n_results: int = 5)`
語義搜索知識庫，回傳最相關的文字片段。

**回傳格式：**
```json
[
  {
    "content": "相關文字片段...",
    "source": "backpack.pdf",
    "page": 3,
    "score": 0.85
  }
]
```

### `list_documents()`
列出所有已索引文件。

**回傳格式：**
```json
[
  {
    "name": "backpack.pdf",
    "type": "pdf",
    "chunks": 10,
    "added_at": "2026-03-25T10:00:00"
  }
]
```

### `delete_document(document_name: str)`
從 RAG 移除指定文件及其所有 chunks。

## REST API Endpoints

| Method | Path | 說明 |
|--------|------|------|
| GET | `/` | 靜態 Web UI |
| POST | `/api/upload` | 上傳文件 (multipart/form-data) |
| GET | `/api/documents` | 列出已索引文件 |
| DELETE | `/api/documents/{name}` | 刪除文件 |
| POST | `/api/search` | 搜索 (JSON body: {query, n_results}) |

## Document Processing Pipeline

1. **接收檔案** → 存到 `uploads/`
2. **格式判斷**：
   - PDF → pdfplumber 擷取每頁文字
   - 圖片 (JPG/PNG/WEBP) → pytesseract OCR
3. **切塊** → RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
4. **Embedding** → Ollama `nomic-embed-text` (768 維向量)
5. **儲存** → ChromaDB collection，metadata 含 source 檔名、頁碼、時間

## Web UI

單頁應用：
- 頂部：拖放上傳區域（支援多檔案）
- 中間：已索引文件列表 + 刪除按鈕
- 底部：搜索測試區（輸入查詢 → 顯示相關片段）
- 中文 UI，深色主題

## OpenClaw Integration

在 openclaw.json 的 agent 設定中加入 MCP server：
```json
{
  "mcpServers": {
    "rag": {
      "command": "python3",
      "args": ["/Users/mike/Desktop/rag/mcp_server.py"]
    }
  }
}
```

## Prerequisites

- Python 3.9+
- Ollama 已安裝並執行 (`ollama pull nomic-embed-text`)
- Tesseract OCR (`brew install tesseract`)

## Constraints

- 完全免費，無雲端 API 費用
- 所有資料本地處理和儲存
- 單機運行，不需要 Docker
