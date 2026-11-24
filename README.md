# ForgeAI – The Agent That Builds Agents

## Overview
ForgeAI is a minimal end‑to‑end system that can **interview** a user, **ground** the request against uploaded documents, **generate** a custom AI agent, **evaluate** it, **optimize** through a critic loop, and finally **run** the agent in a chat UI.

It demonstrates the nine mandatory requirements from the OceanAI assignment using the following tech stack:

- **Python** + **FastAPI** – backend API
- **LangChain** & **LangGraph** – LLM & workflow orchestration
- **Ollama** – local LLM (e.g., `llama3`)
- **ChromaDB** – vector store for RAG
- **PyMuPDF** – PDF ingestion
- **React + Vite** – lightweight frontend dashboard

The project is intentionally simple, readable and runs locally on macOS.

## Folder Structure
```
ForgeAI/
├─ backend/
│   ├─ main.py               # FastAPI entry point
│   ├─ config.py              # Loads .env configuration
│   ├─ ingestion.py           # PDF → chunks → embeddings → Chroma
│   ├─ rag.py                 # Retrieval‑augmented generation helpers
│   ├─ interviewer.py         # Intake & clarification workflow
│   ├─ groundability.py       # Checks for unsupported topics
│   ├─ agent_generator.py     # Generates system prompt & tool list
│   ├─ evaluator.py           # Generates evaluation questions
│   ├─ judge.py               # Scores answers
│   ├─ optimizer.py           # Critic‑optimizer loop (≤3 iters)
│   ├─ agent_runner.py        # Generic chat endpoint for generated agents
│   ├─ tools.py               # Simple tool stubs (e.g., calculator)
│   ├─ prompts/               # Prompt templates (optional)
│   ├─ generated_agents/      # Saved agent definitions
│   ├─ demo_runs/             # Demo artefacts (requirements, logs…)
│   └─ requirements.txt       # Python dependencies
├─ frontend/
│   ├─ src/
│   │   ├─ App.jsx
│   │   └─ index.jsx
│   └─ package.json           # React + Vite deps
├─ .env.example               # Example environment variables
└─ README.md                  # This file
```

## Getting Started
### Prerequisites
- **Python 3.13** (or newer) – `python3 --version`
- **Node.js 20+** – `node --version`
- **Ollama** installed and a model pulled, e.g. `ollama pull llama3`
- **Git** (optional, for cloning the repo)

### Backend Setup
```bash
cd ForgeAI/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# copy example env and edit if needed
cp .env.example .env
uvicorn main:app --reload
```
The API will be reachable at `http://127.0.0.1:8000`.

### Frontend Setup
```bash
cd ../../frontend
npm install
npm run dev
```
The dashboard will be served (by Vite) at `http://localhost:5173` and will communicate with the backend.

## Running the Demonstrations
Two demo scenarios are provided in `backend/demo_runs/`:

1. **HR Policy Assistant** – answers employee‑policy questions.
2. **Ship‑Spare Procurement Assistant** – helps with ordering spare parts for vessels.

To run a demo:
```bash
# 1️⃣ Upload the brief & PDF via the frontend "Create Agent" page.
# 2️⃣ The system will interview, generate, evaluate and optimise the agent.
# 3️⃣ Open the "Agent Chat Playground" tab and start chatting.
# 4️⃣ Results (evaluation scores, logs, final report) are saved under
#    backend/demo_runs/<demo_name>/.
```
All steps are automated – you only need to provide the initial brief and supporting documents.

## Important Notes
- **No paid APIs** are used; the LLM runs locally via Ollama.
- The implementation favours clarity over performance – each module is a thin wrapper around LangChain primitives.
- The optimiser performs up to **three** critique iterations, storing iteration logs in `demo_runs/<demo>/optimizer/`.
- The frontend is deliberately minimal – it provides the required pages without extra styling.

## License
MIT – feel free to fork, extend and experiment.
