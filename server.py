"""FastAPI server: REST API + static Web UI."""

import os
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingest import ingest_file, search_documents, list_documents, delete_document

app = FastAPI(title="RAG Document Manager")

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
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
    file_path = os.path.join(UPLOAD_DIR, name)
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"deleted": name}


@app.post("/api/search")
async def search(req: SearchRequest):
    return search_documents(req.query, req.n_results)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
