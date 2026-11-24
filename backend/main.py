import os
from fastapi import FastAPI, UploadFile, File, Form
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

app = FastAPI()

@app.post("/create-agent")
async def create_agent(
    brief: str = Form(...),
    files: list[UploadFile] = File([])
):
    # 1. Ingest documents
    doc_texts = await ingest_documents(files)
    # 2. Interview for clarification
    requirements = interview_brief(brief, doc_texts)
    # 3. Groundability guard – validate only the extracted requirement fields
    req_fields = requirements.get("requirements", {})
    unsupported = check_groundability(req_fields, doc_texts)
    if unsupported:
        return JSONResponse({"error": "Unsupported topics detected", "topics": unsupported}, status_code=400)
    # 4. Generate agent config
    agent_cfg = generate_agent_config(requirements, doc_texts)
    # 5. Generate evaluation set
    eval_qs = generate_evaluation_questions(agent_cfg)
    # 6. Run initial evaluation (dummy answers)
    dummy_answers = {q: "" for q in eval_qs}
    scores = evaluate_answers(agent_cfg, dummy_answers)
    # 7. Optimize agent (up to 3 iterations)
    optimized_cfg, logs = optimize_agent(agent_cfg, eval_qs)
    # Save artifacts
    demo_dir = Path("backend/demo_runs") / requirements.get("demo_name", "demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "requirements.json").write_text(str(requirements))
    (demo_dir / "agent_config.json").write_text(str(optimized_cfg))
    (demo_dir / "evaluation_questions.json").write_text(str(eval_qs))
    (demo_dir / "optimization_logs.json").write_text(str(logs))
    return {"message": "Agent created and optimized", "demo_path": str(demo_dir)}

@app.post("/chat/{demo_name}")
async def chat(demo_name: str, user_input: str = Form(...)):
    cfg_path = Path("backend/demo_runs") / demo_name / "agent_config.json"
    if not cfg_path.exists():
        return JSONResponse({"error": "Demo not found"}, status_code=404)
    agent_cfg = eval(cfg_path.read_text())  # simple eval for demo
    response = chat_with_agent(agent_cfg, user_input)
    return {"response": response}
