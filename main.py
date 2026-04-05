from fastapi import FastAPI, UploadFile, File
import shutil #handles file operations (copying uploaded files to a specific location)
import os

from rag.loader import load_pdf, chunk_text

from rag.rag_pipeline import build_index, query_rag

app = FastAPI()

UPLOAD_PATH = "data/uploaded.pdf"

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    with open(UPLOAD_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = load_pdf(UPLOAD_PATH)
    chunks = chunk_text(text)

    build_index(chunks)

    return {"message": "PDF processed successfully!"}


@app.get("/ask/")
def ask_question(q: str):
    answer = query_rag(q)
    return {"response": answer}