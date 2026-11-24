import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from .config import settings


def get_retriever() -> Chroma:
    embedding = FakeEmbeddings(size=768)
    vectordb = Chroma(persist_directory=settings.CHROMA_DIR, embedding_function=embedding)
    return vectordb

def retrieve(query: str, top_k: int = 4) -> List[Document]:
    db = get_retriever()
    results = db.similarity_search(query, k=top_k)
    return results
