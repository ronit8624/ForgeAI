import json
from typing import List, Dict
from .config import get_gemini_llm

# Initialize the selected LLM (Groq or Gemini)
llm = get_gemini_llm()

# Import HumanMessage for chat model invocation
try:
    from langchain.schema import HumanMessage
except Exception:
    from langchain_core.messages import HumanMessage


# Initialize Gemini LLM
llm = get_gemini_llm()

def interview_brief(brief: str, docs: List[str]) -> Dict:
    """Generate structured requirements and up to 5 clarification questions.
    The function sends the user brief and a short summary of uploaded documents to the LLM.
    The LLM is asked to return a JSON object with two fields:
        - "requirements": a dict of key/value pairs extracted from the brief.
        - "clarifications": a list of up to 5 short clarification questions.
    The returned dict also contains the original brief and a generated demo name.
    """
    # Summarise documents (first 200 chars each) to give context
    doc_summary = "\n".join([d[:200].replace("\n", " ") for d in docs])
    prompt = f"You are an AI system that extracts concrete requirements from a user's brief and proposes up to five clarification questions. Use the provided document summary for context.\n\nUser brief:\n{brief}\n\nDocument summary (may be empty):\n{doc_summary}\n\nReturn a compact JSON object with keys:\n  \"requirements\": {{key: value}} pairs,\n  \"clarifications\": [list of up to 5 questions].\nMake sure the JSON is parseable."
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        data = json.loads(response)
    except Exception:
        data = {}
    requirements = data.get("requirements", {"brief": brief})
    clarifications = data.get("clarifications", [])[:5]
    demo_name = brief.split("\n")[0].strip().replace(" ", "_")[:30]
    return {
        "brief": brief,
        "demo_name": demo_name or "demo",
        "requirements": requirements,
        "clarifications": clarifications,
        "documents_summary": doc_summary,
    }
