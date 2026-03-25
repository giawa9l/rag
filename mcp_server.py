"""MCP server for OpenClaw - exposes RAG tools via stdio."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
