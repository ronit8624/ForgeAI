import json
import re
from typing import Dict, List
from langchain_core.messages import HumanMessage
from .config import get_gemini_llm

llm = get_gemini_llm()


def generate_agent_config(requirements: Dict, docs: List[str]) -> Dict:
    """Generate a rich agent configuration using Groq."""
    brief = requirements.get("brief", "")
    req_fields = requirements.get("requirements", {})
    doc_snippets = "\n".join([d[:400].replace("\n", " ") for d in docs])

    prompt = (
        "You are designing an AI agent. Based on the brief and document context below, "
        "generate a detailed agent configuration as a JSON object with these keys:\n"
        '  "name": short agent name,\n'
        '  "system_prompt": detailed system prompt (3-5 sentences),\n'
        '  "scope": list of supported topics,\n'
        '  "audience": intended users,\n'
        '  "tone": response tone (e.g. professional, friendly),\n'
        '  "supported_topics": list of topics from the documents,\n'
        '  "refusal_rules": when to refuse,\n'
        '  "tools": ["calculator"] if math is needed else [],\n'
        '  "output_format": "text"\n\n'
        f"Brief: {brief}\n"
        f"Requirements: {json.dumps(req_fields)}\n"
        f"Document context:\n{doc_snippets}\n\n"
        "Return ONLY the JSON object."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content if hasattr(response, "content") else str(response)

    # Extract JSON
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        cfg = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        cfg = json.loads(match.group()) if match else {}

    # Ensure mandatory keys exist
    cfg.setdefault("name", requirements.get("demo_name", "agent"))
    cfg.setdefault("system_prompt", f"You are an AI assistant for: {brief}. Use the provided documents for grounding.")
    cfg.setdefault("use_rag", True)
    cfg.setdefault("tools", [])
    cfg["documents"] = docs  # always store raw docs for chat runner

    return cfg
