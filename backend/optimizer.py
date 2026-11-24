import json
from typing import Dict, List, Tuple
from .config import get_gemini_llm

llm = get_gemini_llm()
from .evaluator import generate_evaluation_questions
from .judge import evaluate_answers
from .agent_generator import generate_agent_config
from .agent_runner import chat_with_agent
from .tools import calculator
import re


def optimize_agent(agent_cfg: Dict, eval_questions: List[str], max_iters: int = 3) -> Tuple[Dict, List[Dict], str]:
    """Run a genuine critic‑optimizer loop.
    For each iteration we:
    1️⃣ Generate answers to the evaluation questions using the current agent.
    2️⃣ Evaluate those answers with the LLM judge.
    3️⃣ Ask the LLM for a critique of the *configuration* given the evaluation results.
    4️⃣ Apply simple rule‑based fixes inferred from the critique (e.g., add missing tools).
    The loop stops after `max_iters` or when the average score reaches ≥ 0.85.
    Returns the final configuration, a list of iteration logs, and a final report string.
    """
    logs: List[Dict] = []
    cfg = agent_cfg.copy()
    # Prepare a simple tool‑lookup for possible fixes
    available_tools = {"calculator": calculator}

    for i in range(max_iters):
        # 1️⃣ Generate answers
        answers = {q: chat_with_agent(cfg, q) for q in eval_questions}
        # 2️⃣ Evaluate answers
        scores = evaluate_answers(cfg, answers)
        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        # 3️⃣ Prompt LLM for critique
        critique_prompt = (
            "You are reviewing an AI agent configuration. The configuration is given as JSON below.\n"
            f"Configuration: {json.dumps(cfg, ensure_ascii=False)}\n"
            "Evaluation results (question -> score):\n" + json.dumps(scores, ensure_ascii=False) + "\n"
            "Provide a concise critique (max two sentences) and suggest concrete fixes.\n"
            "If a tool is missing that would improve performance, mention its name."
        )
        try:
            critique_resp = llm.invoke(critique_prompt)
            critique = critique_resp.strip()
        except Exception:
            critique = "No clear critique obtained."
        # 4️⃣ Apply simple rule‑based fixes based on critique text
        for tool_name in available_tools.keys():
            if tool_name in critique.lower() and tool_name not in cfg.get("tools", []):
                cfg.setdefault("tools", []).append(tool_name)
                critique += f" Added missing {tool_name} tool."
        logs.append({
            "iteration": i + 1,
            "avg_score": round(avg_score, 3),
            "critique": critique,
            "config": cfg,
        })
        if avg_score >= 0.85:
            break
    # Generate final report
    report_lines = ["# Optimization Report", ""]
    report_lines.append(f"**Final average score:** {round(avg_score,3)}")
    report_lines.append("\n## Iteration details")
    for entry in logs:
        report_lines.append(f"### Iteration {entry['iteration']}")
        report_lines.append(f"* Avg score: {entry['avg_score']}")
        report_lines.append(f"* Critique: {entry['critique']}")
        report_lines.append("* Config snapshot: ```" + json.dumps(entry['config'], ensure_ascii=False, indent=2) + "```\n")
    report = "\n".join(report_lines)
    return cfg, logs, report
