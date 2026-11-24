from typing import Dict
from .config import get_gemini_llm
from .rag import retrieve
from langchain_core.messages import HumanMessage, SystemMessage

llm = get_gemini_llm()


def chat_with_agent(agent_cfg: Dict, user_input: str) -> str:
    """Answer a question using the agent's system prompt and RAG-retrieved documents."""
    system_prompt = agent_cfg.get("system_prompt", "You are a helpful AI assistant.")
    use_rag = agent_cfg.get("use_rag", False)

    top_k = agent_cfg.get("top_k", 4)
    docs = retrieve(user_input, top_k=top_k) if use_rag else []
    context = "\n\n---\n\n".join([d.page_content for d in docs]) if docs else ""

    messages = [SystemMessage(content=system_prompt)]
    if context:
        messages.append(HumanMessage(content=f"Context from documents:\n{context}\n\nQuestion: {user_input}"))
    else:
        messages.append(HumanMessage(content=user_input))

    try:
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Error generating response: {e}"
