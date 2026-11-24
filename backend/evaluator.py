import json
from typing import List, Dict
from .config import settings
from .rag import retrieve
from .config import get_gemini_llm

# Initialize Gemini LLM
llm = get_gemini_llm()

def generate_evaluation_questions(agent_cfg: Dict) -> List[str]:
    """Generate at least 20 evaluation questions based on the agent's purpose and uploaded documents.
    The function extracts the brief and a short slice of each document, then asks the LLM to produce a numbered list.
    Returns a plain list of question strings.
    """
    brief = agent_cfg.get("system_prompt", "")
    docs = agent_cfg.get("documents", [])
    doc_snippets = "\n".join([d[:300].replace("\n", " ") for d in docs])
    prompt = (
        "You are to create a set of at least 20 distinct evaluation questions that thoroughly test an AI agent.\n"
        "The agent's purpose is described in the system prompt below, and the following document snippets provide domain context.\n"
        "Create a JSON array of strings, each a single clear question. Include a mix of:\n"
        "- straightforward factual queries that should be answerable using the documents,\n"
        "- out‑of‑scope queries,\n"
        "- edge‑case or ambiguous queries that test the agent's clarification behaviour.\n"
        "System prompt:\n" + brief + "\n\nDocument snippets (may be empty):\n" + doc_snippets + "\n\nRespond only with JSON.")
    response = llm.invoke(prompt)
    try:
        questions = json.loads(response)
        if isinstance(questions, list) and len(questions) >= 20:
            return questions[:20]
        # Fallback: if not enough, slice or pad with generic ones
        return (questions if isinstance(questions, list) else [])[:20]
    except Exception:
        # Very simple fallback: generate generic numbered questions
        return [f"Question {i+1}: Please describe the agent's behaviour for scenario {i+1}." for i in range(20)]
