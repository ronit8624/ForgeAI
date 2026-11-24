import fitz  # PyMuPDF
from fastapi import UploadFile
from pathlib import Path
from typing import List

from .config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings


async def ingest_documents(files: List[UploadFile]) -> List[str]:
    """Extract text from PDFs, chunk, embed (FakeEmbeddings) and store in ChromaDB.

    NOTE: FakeEmbeddings are used for local testing – similarity search will
    return documents but NOT by semantic relevance. This is a known limitation
    disclosed to the user. For production, replace with a real embedding model.
    """
    texts = []
    for upload in files:
        if Path(upload.filename).suffix.lower() != ".pdf":
            continue
        data = await upload.read()
        doc = fitz.open(stream=data, filetype="pdf")
        full_text = "".join([page.get_text() for page in doc])
        doc.close()
        texts.append(full_text)

    if not texts:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for txt in texts:
        chunks.extend(splitter.create_documents([txt]))

    # Store in ChromaDB (FakeEmbeddings – see NOTE above)
    embedding = FakeEmbeddings(size=768)
    Chroma.from_documents(chunks, embedding, persist_directory=settings.CHROMA_DIR)
    # Note: persist() is deprecated in Chroma ≥0.4; data is auto-persisted.
    return texts
