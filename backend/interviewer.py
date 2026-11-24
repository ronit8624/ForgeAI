import json
import re
from typing import List, Dict
from .config import get_gemini_llm
from langchain_core.messages import HumanMessage

# Initialize LLM once at module load (Groq if configured)
llm = get_gemini_llm()


def _parse_json(text: str) -> dict:
    """Try to extract a JSON object from LLM output even if wrapped in markdown."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # Try to find first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}


def interview_brief(brief: str, docs: List[str]) -> Dict:
    """Generate structured requirements and up to 5 clarification questions using Groq."""
    doc_summary = "\n".join([d[:300].replace("\n", " ") for d in docs])
    prompt = (
        "You are an AI system that extracts concrete requirements from a user's brief.\n"
        "Return ONLY a valid JSON object with these exact keys:\n"
        '  "requirements": {key: value pairs of concrete requirements},\n'
        '  "clarifications": [list of up to 5 short clarification questions]\n\n'
        f"User brief:\n{brief}\n\n"
        f"Document summary (may be empty):\n{doc_summary}\n\n"
        "Return ONLY the JSON object, no markdown, no explanation."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    # Extract text content from chat model response
    text = response.content if hasattr(response, "content") else str(response)
    data = _parse_json(text)

    requirements = data.get("requirements", {"topic": brief[:60]})
    clarifications = data.get("clarifications", [])[:5]
    demo_name = re.sub(r"[^a-zA-Z0-9_-]", "_", brief[:40]).strip("_")

    return {
        "brief": brief,
        "demo_name": demo_name or "demo",
        "requirements": requirements,
        "clarifications": clarifications,
        "documents_summary": doc_summary,
    }
