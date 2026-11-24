import json
import re
from typing import Dict, List
from .config import get_gemini_llm
from .rag import retrieve
from langchain_core.messages import HumanMessage

llm = get_gemini_llm()


def _extract_score(text: str) -> float:
    """Extract numeric score from LLM response."""
    # Try JSON first
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return float(data.get("score", 0.5))
    except Exception:
        pass
    # Try regex for a number 0-1 in the text
    match = re.search(r"(?:score[:\s]+)?([01](?:\.\d+)?)", cleaned)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            pass
    return 0.5


def evaluate_answers(agent_cfg: Dict, answers: Dict[str, str]) -> Dict[str, float]:
    """Evaluate each answer using the LLM judge."""
    scores: Dict[str, float] = {}
    use_rag = agent_cfg.get("use_rag", False)

    for q, ans in answers.items():
        if not ans or ans.strip() == "":
            # Empty answer - cannot evaluate fairly
            scores[q] = 0.0
            continue

        docs = retrieve(q) if use_rag else []
        doc_text = "\n---\n".join([d.page_content for d in docs])

        eval_prompt = (
            "You are evaluating an AI assistant's answer.\n"
            f"Question: {q}\n"
            f"Answer: {ans}\n"
            + (f"Relevant documents:\n{doc_text}\n" if doc_text else "")
            + "Score the answer 0.0 to 1.0: 1.0 = fully correct and grounded, "
            "0.5 = partially correct, 0.0 = wrong or hallucinated.\n"
            'Respond ONLY with JSON: {"score": <number>}'
        )
        try:
            resp = llm.invoke([HumanMessage(content=eval_prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            score = _extract_score(text)
        except Exception:
            # Simple deterministic fallback heuristic only when LLM fails
            if "don't know" in ans.lower() or "cannot" in ans.lower():
                score = 0.3  # refusal – could be right or wrong depending on context
            else:
                score = 0.5
        scores[q] = round(min(max(score, 0.0), 1.0), 2)
    return scores
