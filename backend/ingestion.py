import fitz
from fastapi import UploadFile
from fastapi import UploadFile
import asyncio
from pathlib import Path
from typing import List

from .config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

async def ingest_documents(files: List[UploadFile]):
    """Read uploaded PDFs, split into chunks, embed into ChromaDB.
    Returns list of all extracted texts (for later use)."""
    texts = []
    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix != ".pdf":
            continue
        data = await upload.read()
        doc = fitz.open(stream=data, filetype="pdf")
        full_text = "".join([page.get_text() for page in doc])
        texts.append(full_text)
    if not texts:
        return []
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for txt in texts:
        docs.extend(splitter.create_documents([txt]))
    # Store in Chroma
    embedding = FakeEmbeddings(size=768)
    vectordb = Chroma.from_documents(docs, embedding, persist_directory=settings.CHROMA_DIR)
    vectordb.persist()
    return texts
