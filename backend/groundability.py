import json
import re
from typing import List, Dict
from langchain_core.messages import HumanMessage
from .config import get_gemini_llm

llm = get_gemini_llm()


def _parse_json(text: str) -> dict:
    """Extract a JSON object from LLM output."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}


def check_groundability(requirements: Dict, docs: List[str]) -> List[str]:
    """Separate behavioral/meta instructions from factual domain requirements,
    and verify whether factual domain requirements are supported by the uploaded documents.

    Returns a list of genuinely unsupported factual topic keys/descriptions.
    """
    if not requirements:
        return []

    if not docs or not any(d.strip() for d in docs):
        return ["no_uploaded_documents"]

    doc_text = "\n\n---\n\n".join(d[:4000] for d in docs)

    prompt = (
        "You are a Groundability Guard verifying whether an AI agent's requirements "
        "can be supported by the uploaded documents.\n\n"
        "IMPORTANT RULES:\n"
        "1. Behavioral/operational instructions and metadata (such as tone, response style, "
        "knowledge_source, fallback_behavior, refusal rules, output format, language, "
        "audience, or general task framing like 'answer questions using the document') "
        "are NOT factual claims and must NEVER be flagged as unsupported.\n"
        "2. Only flag a requirement if it asks for specific factual/domain knowledge or topics "
        "that are clearly NOT covered anywhere in the uploaded documents.\n\n"
        f"Extracted Requirements:\n{json.dumps(requirements, indent=2)}\n\n"
        f"Uploaded Document Content:\n{doc_text}\n\n"
        'Return ONLY a JSON object with a single key "unsupported_topics" containing a list '
        'of strings (e.g. {"unsupported_topics": []}). If all factual topics are supported, '
        'return {"unsupported_topics": []}.'
    )

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        data = _parse_json(text)
        unsupported = data.get("unsupported_topics", [])
        if isinstance(unsupported, list):
            return [str(item) for item in unsupported if item]
    except Exception:
        pass

    return []
