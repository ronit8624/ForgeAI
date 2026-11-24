import json
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path

from .config import settings
from .ingestion import ingest_documents
from .interviewer import interview_brief
from .groundability import check_groundability
from .agent_generator import generate_agent_config
from .evaluator import generate_evaluation_questions
from .judge import evaluate_answers
from .optimizer import optimize_agent
from .agent_runner import chat_with_agent

app = FastAPI(title="ForgeAI – The Agent That Builds Agents")

# Allow frontend (Vite default port 5173) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/create-agent")
async def create_agent(
    brief: str = Form(...),
    files: list[UploadFile] = File(default=[])
):
    try:
        # 1. Ingest documents
        doc_texts = await ingest_documents(files)

        # 2. Interview for structured requirements
        requirements = interview_brief(brief, doc_texts)

        # 3. Groundability guard – only check extracted requirement fields
        req_fields = requirements.get("requirements", {})
        unsupported = check_groundability(req_fields, doc_texts)
        if unsupported:
            return JSONResponse(
                {"error": "Unsupported topics detected", "topics": unsupported},
                status_code=400
            )

        # 4. Generate agent configuration
        agent_cfg = generate_agent_config(requirements, doc_texts)

        # 5. Generate evaluation set (≥20 questions)
        eval_qs = generate_evaluation_questions(agent_cfg)

        # 6. Optimize agent (up to 3 iterations), returns (config, logs)
        optimized_cfg, logs = optimize_agent(agent_cfg, eval_qs)

        # 7. Run final evaluation with real answers for the report
        final_answers = {q: chat_with_agent(optimized_cfg, q) for q in eval_qs[:5]}
        final_scores = evaluate_answers(optimized_cfg, final_answers)

        # 8. Save artifacts
        demo_dir = Path("backend/demo_runs") / requirements.get("demo_name", "demo")
        demo_dir.mkdir(parents=True, exist_ok=True)

        (demo_dir / "requirements.json").write_text(
            json.dumps(requirements, indent=2, ensure_ascii=False)
        )
        # Save config without large document blobs for readability
        cfg_to_save = {k: v for k, v in optimized_cfg.items() if k != "documents"}
        (demo_dir / "agent_config.json").write_text(
            json.dumps(cfg_to_save, indent=2, ensure_ascii=False)
        )
        (demo_dir / "evaluation_questions.json").write_text(
            json.dumps(eval_qs, indent=2, ensure_ascii=False)
        )
        (demo_dir / "optimization_logs.json").write_text(
            json.dumps(logs, indent=2, ensure_ascii=False)
        )
        (demo_dir / "final_scores.json").write_text(
            json.dumps(final_scores, indent=2, ensure_ascii=False)
        )

        return {
            "message": "Agent created and optimized",
            "demo_path": str(demo_dir),
            "demo_name": requirements.get("demo_name"),
            "evaluation_question_count": len(eval_qs),
            "optimization_iterations": len(logs),
            "final_avg_score": round(sum(final_scores.values()) / len(final_scores), 3) if final_scores else None,
            "clarifications": requirements.get("clarifications", []),
        }

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)


@app.post("/chat/{demo_name}")
async def chat(demo_name: str, user_input: str = Form(...)):
    cfg_path = Path("backend/demo_runs") / demo_name / "agent_config.json"
    docs_path = Path("backend/demo_runs") / demo_name / "requirements.json"

    if not cfg_path.exists():
        return JSONResponse({"error": f"Demo '{demo_name}' not found"}, status_code=404)

    agent_cfg = json.loads(cfg_path.read_text())

    # Reload document texts from requirements for RAG
    if docs_path.exists():
        req = json.loads(docs_path.read_text())
        agent_cfg["documents"] = []  # Docs were not saved (too large) - RAG still works via Chroma
    agent_cfg["use_rag"] = True

    response = chat_with_agent(agent_cfg, user_input)
    return {"response": response, "demo_name": demo_name}


@app.get("/health")
def health():
    return {"status": "ok", "provider": settings.LLM_PROVIDER, "model": settings.GROQ_MODEL}
