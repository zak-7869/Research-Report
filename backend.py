import os
import json
from typing import TypedDict, List, Dict, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is required")

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.2,
    max_retries=2,
)

class ResearchState(TypedDict, total=False):
    topic: str
    search_queries: List[str]
    sources: List[Dict[str, Any]]
    research: str
    report: str


def _text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def research_agent(state: ResearchState) -> Dict[str, Any]:
    topic = state["topic"]
    planner = llm.invoke([
        SystemMessage(content=(
            "You are a research planning agent. Create 3 focused web-search queries "
            "for the user's topic. Prefer primary sources, official docs, reputable "
            "news or research organizations. Return ONLY a JSON array of strings."
        )),
        HumanMessage(content=topic),
    ])

    raw = _text(planner).strip()
    try:
        queries = json.loads(raw)
        if not isinstance(queries, list):
            raise ValueError
        queries = [str(q) for q in queries[:3]]
    except Exception:
        queries = [topic]

    sources: List[Dict[str, Any]] = []
    if TAVILY_API_KEY:
        async_client = httpx.Client(timeout=30)
        try:
            for query in queries:
                response = async_client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": TAVILY_API_KEY,
                        "query": query,
                        "search_depth": "advanced",
                        "max_results": 5,
                        "include_answer": False,
                    },
                )
                response.raise_for_status()
                data = response.json()
                for item in data.get("results", []):
                    sources.append({
                        "title": item.get("title", "Untitled"),
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                    })
        finally:
            async_client.close()
    else:
        sources = [{
            "title": "Web search not configured",
            "url": "",
            "content": "TAVILY_API_KEY is missing. The report will be generated from the model's existing knowledge and clearly marked accordingly.",
        }]

    # De-duplicate URLs and keep the context bounded.
    seen = set()
    unique_sources = []
    for s in sources:
        url = s.get("url")
        key = url or s.get("title")
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)
    unique_sources = unique_sources[:12]

    evidence = "\n\n".join(
        f"SOURCE {i+1}\nTitle: {s['title']}\nURL: {s['url']}\nExtract: {s['content']}"
        for i, s in enumerate(unique_sources)
    )

    synthesis = llm.invoke([
        SystemMessage(content=(
            "You are a research agent. Analyze the supplied web evidence for the topic. "
            "Separate established facts from uncertainty. Do not invent citations. "
            "Produce concise research notes with key findings, caveats, and source URLs."
        )),
        HumanMessage(content=f"TOPIC:\n{topic}\n\nWEB EVIDENCE:\n{evidence}"),
    ])

    return {
        "search_queries": queries,
        "sources": unique_sources,
        "research": _text(synthesis),
    }


def report_agent(state: ResearchState) -> Dict[str, Any]:
    topic = state["topic"]
    research = state.get("research", "")
    sources = state.get("sources", [])
    source_list = "\n".join(f"- {s['title']} — {s['url']}" for s in sources if s.get("url"))

    report = llm.invoke([
        SystemMessage(content=(
            "You are a report-writing agent. Turn research notes into a polished, "
            "well-structured report in Markdown. Use headings, bullets, and a concise "
            "executive summary. Be factual and transparent about uncertainty. "
            "Include a Sources section using ONLY the supplied URLs."
        )),
        HumanMessage(content=(
            f"TOPIC:\n{topic}\n\nRESEARCH NOTES:\n{research}\n\n"
            f"SOURCES:\n{source_list}"
        )),
    ])
    return {"report": _text(report)}


graph = StateGraph(ResearchState)
graph.add_node("research_agent", research_agent)
graph.add_node("report_agent", report_agent)
graph.add_edge(START, "research_agent")
graph.add_edge("research_agent", "report_agent")
graph.add_edge("report_agent", END)
app_graph = graph.compile()

app = FastAPI(title="LangGraph Groq Research Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: str

@app.get("/api/health")
def health():
    return {"ok": True, "model": GROQ_MODEL, "web_search": bool(TAVILY_API_KEY)}

@app.post("/api/research")
def run_research(request: ResearchRequest):
    topic = request.topic.strip()
    if len(topic) < 3:
        raise HTTPException(status_code=400, detail="Topic must be at least 3 characters.")
    if len(topic) > 500:
        raise HTTPException(status_code=400, detail="Topic is too long.")
    try:
        result = app_graph.invoke({"topic": topic})
        return {
            "topic": topic,
            "queries": result.get("search_queries", []),
            "research": result.get("research", ""),
            "report": result.get("report", ""),
            "sources": result.get("sources", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# Serve the single-page frontend.
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def index():
    return FileResponse("frontend/index.html")



