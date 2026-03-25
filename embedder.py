"""Ollama embedding wrapper for nomic-embed-text."""

import httpx

OLLAMA_BASE = "http://localhost:11434"
MODEL = "nomic-embed-text"


def get_embeddings(texts: list) -> list:
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


def get_embedding(text: str) -> list:
    """Get embedding for a single text."""
    return get_embeddings([text])[0]
