#!/bin/bash
# RAG Document Manager - 一鍵啟動
cd "$(dirname "$0")"
source venv/bin/activate
echo "🚀 RAG Server 啟動中..."
echo "📂 開啟瀏覽器: http://localhost:8000"
open http://localhost:8000 &
python server.py
