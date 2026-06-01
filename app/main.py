from typing import Any
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.graph import ask_audit_rag


BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"


app = FastAPI(
    title="Audit Knowledge RAG API",
    description="基于 Qdrant + LangGraph + 本地/自建大模型的审计知识问答系统",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "audit-rag",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = ask_audit_rag(req.question)
    return result
