# LangGraph + Groq Research → Report

A minimal two-agent research application:

1. **Research agent** — uses Groq to create focused search queries, Tavily to collect web evidence, then Groq to synthesize research notes.
2. **Report agent** — uses Groq to turn the research notes and source URLs into a structured Markdown report.
3. **Frontend** — one `index.html`, served directly by FastAPI.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your keys:

```env
GROQ_API_KEY=...
TAVILY_API_KEY=...
GROQ_MODEL=openai/gpt-oss-120b
```

Then run:

```bash
uvicorn backend:app --reload
```

Open http://127.0.0.1:8000

## Architecture

```text
index.html
    │ POST /api/research
    ▼
FastAPI
    │
    ▼
LangGraph StateGraph
    │
    ├── research_agent
    │     ├── Groq: plan queries
    │     ├── Tavily: web search
    │     └── Groq: synthesize evidence
    │
    └── report_agent
          └── Groq: write report



```

## DEPLOY 
   **** VERCEL****

   


If `TAVILY_API_KEY` is absent, the app still runs, but the report is generated without live web search and explicitly receives a missing-search notice.
