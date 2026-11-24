import json
from typing import Dict, List
from .config import settings
from .config import get_gemini_llm

# Initialize Gemini LLM
llm = get_gemini_llm()

def generate_agent_config(requirements: Dict, docs: List[str]) -> Dict:
    """Create a simple agent configuration.
    The config includes a system prompt, RAG flag and a list of tool names.
    """
    system_prompt = f"You are an AI assistant for the task: {requirements.get('brief', '')}. Use the provided documents for grounding."
    return {
        "name": requirements.get("demo_name", "agent"),
        "system_prompt": system_prompt,
        "use_rag": True,
        "tools": ["calculator"],
        "documents": docs,
    }
