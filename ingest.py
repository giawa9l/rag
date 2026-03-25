"""Document ingestion: parse -> chunk -> embed -> store in ChromaDB."""

import os
import uuid
from datetime import datetime, timezone

import chromadb
import pdfplumber
import pytesseract
from PIL import Image

from embedder import get_embeddings, get_embedding

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_collection():
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def extract_text_from_pdf(file_path):
    """Extract text from PDF, one entry per page."""
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": i + 1})
    return pages


def extract_text_from_image(file_path):
    """OCR an image file."""
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img, lang="chi_tra+eng")
    if text.strip():
        return [{"text": text, "page": 1}]
    return []


def chunk_text(text):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def ingest_file(file_path):
    """Ingest a file into ChromaDB. Returns metadata about the ingestion."""
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

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

    texts = [c["text"] for c in all_chunks]
    embeddings = get_embeddings(texts)

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


def search_documents(query, n_results=5):
    """Search documents by semantic similarity."""
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


def list_documents():
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


def delete_document(document_name):
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
