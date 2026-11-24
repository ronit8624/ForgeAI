import json
import re
from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage
from .config import get_gemini_llm
from .agent_runner import chat_with_agent
from .judge import evaluate_answers
from .tools import calculator

llm = get_gemini_llm()

_VALID_CATEGORIES = {"retrieval_miss", "prompt_defect", "hallucination", "wrong_refusal", "none"}


def _parse_critique(text: str) -> Dict:
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return {"failure_category": "prompt_defect", "reason": text[:200], "suggested_fix": ""}


def optimize_agent(agent_cfg: Dict, eval_questions: List[str], max_iters: int = 3) -> Tuple[Dict, List[Dict]]:
    """Critic-optimizer loop (maximum 3 iterations).

    Records iteration number, failure category, changes made, reason,
    actual score before, and actual score after.
    """
    logs: List[Dict] = []
    cfg = dict(agent_cfg)
    available_tools = {"calculator": calculator}

    # Initial evaluation
    answers = {q: chat_with_agent(cfg, q) for q in eval_questions}
    scores = evaluate_answers(cfg, answers)
    current_score = round(sum(scores.values()) / len(scores), 3) if scores else 0.0

    for i in range(max_iters):
        score_before = current_score
        if score_before >= 0.85:
            logs.append({
                "iteration": i + 1,
                "failure_category": "none",
                "changes_made": [],
                "reason": "Average score already meets target threshold (>= 0.85).",
                "score_before": score_before,
                "score_after": score_before,
            })
            break

        # Ask LLM critic to classify failure and propose a targeted fix
        low_scoring = {q[:80]: {"answer": answers[q][:120], "score": s} for q, s in scores.items() if s < 0.8}
        critique_prompt = (
            "You are a critic reviewing an AI agent configuration and its failed test cases.\n"
            f"System prompt: {cfg.get('system_prompt', '')}\n"
            f"Tools: {cfg.get('tools', [])}\n"
            f"Average score: {score_before}\n"
            f"Low scoring cases (sample): {json.dumps(dict(list(low_scoring.items())[:8]))}\n\n"
            "Classify the primary failure category as ONE of: "
            "retrieval_miss, prompt_defect, hallucination, wrong_refusal, none.\n"
            "Return ONLY a JSON object with keys:\n"
            '  "failure_category": "<category>",\n'
            '  "reason": "<brief explanation>",\n'
            '  "suggested_fix": "<concrete prompt or tool fix>"'
        )
        try:
            resp = llm.invoke([HumanMessage(content=critique_prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            critique = _parse_critique(text)
        except Exception as e:
            critique = {"failure_category": "none", "reason": f"Critique error: {e}", "suggested_fix": ""}

        category = str(critique.get("failure_category", "none")).lower().strip()
        if category not in _VALID_CATEGORIES:
            category = "prompt_defect"
        reason = str(critique.get("reason", ""))
        suggested_fix = str(critique.get("suggested_fix", ""))

        # Apply targeted fix
        changes_made: List[str] = []
        if category == "hallucination":
            cfg["system_prompt"] = (
                cfg.get("system_prompt", "")
                + "\nSTRICT GROUNDING RULE: Answer ONLY using facts explicitly stated in the provided context. "
                "Do not invent or extrapolate any information."
            )
            changes_made.append("Added strict grounding rule to system_prompt")
        elif category == "wrong_refusal":
            cfg["system_prompt"] = (
                cfg.get("system_prompt", "")
                + "\nREFUSAL RULE: If the answer is present in the context, answer directly. "
                "Only refuse when the question is genuinely not covered by the context."
            )
            changes_made.append("Clarified refusal boundary in system_prompt")
        elif category == "retrieval_miss":
            cfg["use_rag"] = True
            cfg["top_k"] = min(cfg.get("top_k", 4) + 2, 8)
            changes_made.append(f"Increased retrieval top_k to {cfg['top_k']}")
        elif category == "prompt_defect":
            if suggested_fix:
                cfg["system_prompt"] = cfg.get("system_prompt", "") + f"\nGUIDANCE: {suggested_fix}"
                changes_made.append(f"Appended targeted guidance to system_prompt: {suggested_fix[:120]}")

        for tool_name in available_tools:
            if tool_name in suggested_fix.lower() and tool_name not in cfg.get("tools", []):
                cfg.setdefault("tools", []).append(tool_name)
                changes_made.append(f"Added tool: {tool_name}")

        # Re-evaluate only if changes were applied
        if changes_made:
            answers = {q: chat_with_agent(cfg, q) for q in eval_questions}
            scores = evaluate_answers(cfg, answers)
            score_after = round(sum(scores.values()) / len(scores), 3) if scores else score_before
        else:
            score_after = score_before

        current_score = score_after
        logs.append({
            "iteration": i + 1,
            "failure_category": category,
            "changes_made": changes_made if changes_made else ["No changes applied"],
            "reason": reason,
            "score_before": score_before,
            "score_after": score_after,
        })

        if not changes_made or score_after >= 0.85:
            break

    return cfg, logs
