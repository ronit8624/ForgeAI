import json
import re
from typing import List, Dict
from .config import get_gemini_llm
from langchain_core.messages import HumanMessage

llm = get_gemini_llm()


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from LLM output that may be wrapped in markdown."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    # Try to find [...] block
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except Exception:
            pass
    return []


def generate_evaluation_questions(agent_cfg: Dict) -> List[str]:
    """Generate at least 20 evaluation questions from the agent's documents and purpose."""
    system_prompt = agent_cfg.get("system_prompt", "")
    docs = agent_cfg.get("documents", [])
    doc_snippets = "\n".join([d[:500].replace("\n", " ") for d in docs])

    prompt = (
        "You are creating evaluation test cases for an AI agent.\n"
        "Agent purpose: " + system_prompt + "\n\n"
        "Document content (excerpt):\n" + doc_snippets + "\n\n"
        "Generate EXACTLY 25 evaluation questions as a JSON array of strings.\n"
        "Include:\n"
        "- 10 factual questions answerable from the document\n"
        "- 5 out-of-scope questions the agent should refuse\n"
        "- 5 edge-case or ambiguous questions\n"
        "- 5 questions testing specific details from the document\n\n"
        "Return ONLY a JSON array of question strings, no explanation, no markdown."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content if hasattr(response, "content") else str(response)
    questions = _extract_json_array(text)

    if len(questions) >= 20:
        return questions[:25]

    # Fallback: generate generic ones only if LLM truly failed
    # (but do NOT silently substitute - report the issue in question text)
    if not questions:
        return [
            f"[FALLBACK - LLM generation failed] Question {i+1}: Describe a key aspect of the agent's domain."
            for i in range(20)
        ]
    # Pad if slightly short
    while len(questions) < 20:
        questions.append(f"What other information from the document is relevant to this agent's purpose?")
    return questions
