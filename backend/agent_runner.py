import json
from typing import Dict
from .config import get_gemini_llm
from .rag import retrieve
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Gemini LLM
llm = get_gemini_llm()

def build_prompt(system_prompt: str, user_input: str, retrieved_docs: list) -> str:
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
    return f"{system_prompt}\n\nUser: {user_input}\n\nContext:\n{context}\n\nAssistant:" if context else f"{system_prompt}\n\nUser: {user_input}\n\nAssistant:"

def chat_with_agent(agent_cfg: Dict, user_input: str) -> str:
    system_prompt = agent_cfg.get("system_prompt", "You are an AI assistant.")
    use_rag = agent_cfg.get("use_rag", False)
    docs = retrieve(user_input) if use_rag else []
    prompt = build_prompt(system_prompt, user_input, docs)
    response = llm.invoke(prompt)
    return response
