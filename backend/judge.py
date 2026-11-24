import json
from typing import Dict, List
from .config import get_gemini_llm
from .rag import retrieve


llm = get_gemini_llm()

def evaluate_answers(agent_cfg: Dict, answers: Dict[str, str]) -> Dict[str, float]:
    """Evaluate each answer using the LLM.
    For each question we retrieve relevant documents (if RAG enabled) and ask the LLM
    to score the answer on correctness, grounding and refusal behaviour.
    The LLM should return a JSON object like {"score": 0.87}.
    If the LLM fails we fall back to a simple heuristic.
    """
    scores: Dict[str, float] = {}
    use_rag = agent_cfg.get("use_rag", False)
    for q, ans in answers.items():
        # Retrieve context for the question if RAG is on
        docs = retrieve(q) if use_rag else []
        doc_text = "\n---\n".join([d.page_content for d in docs])
        eval_prompt = (
            "You are evaluating an AI assistant's answer.\n"
            f"Question: {q}\n"
            f"Answer: {ans}\n"
            f"Relevant documents (if any):\n{doc_text}\n"
            "Score the answer on a scale of 0 to 1 where 1 means fully correct, well‑grounded, and does not refuse when it should answer.\n"
            "Respond ONLY with a JSON object containing a single key \"score\" with the numeric value."
        )
        try:
            resp = llm.invoke(eval_prompt)
            data = json.loads(resp)
            score = float(data.get("score", 0))
        except Exception:
            # Simple fallback heuristic
            score = 0.5 if "don't know" in ans.lower() else 0.8
        scores[q] = round(min(max(score, 0.0), 1.0), 2)
    return scores
