from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "none").lower()  # none | ollama | openai
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(title="Career Counseling Chatbot API", version="1.0.0")

client = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL)


def normalize_collection_name(domain_name: str) -> str:
    s = (domain_name or "").strip().replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in s)
    return f"career_{(s or 'unknown').lower()}"


class Citation(BaseModel):
    doc_id: str
    domain_name: str
    source_spec: Optional[str] = None
    excerpt: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="학생 질문/상담 요청")
    domain: Optional[str] = Field(None, description="데이터 도메인 필터")
    top_k: int = Field(5, ge=1, le=10)
    history: List[Dict[str, str]] = Field(default_factory=list, description="[{role,user|assistant, content}]")


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    used_collection: str


class AskRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    top_k: int = Field(4, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    used_collection: str


def embed_query(q: str) -> List[float]:
    return embedder.encode(["query: " + q], normalize_embeddings=True)[0].tolist()


def collection_exists(name: str) -> bool:
    return any(c.name == name for c in client.get_collections().collections)


def search_chunks(collection: str, qvec: List[float], top_k: int) -> List[Dict[str, Any]]:
    result = client.search(collection_name=collection, query_vector=qvec, limit=top_k, with_payload=True)
    return [{"score": float(r.score), "payload": (r.payload or {})} for r in result]


def build_prompt(question: str, history: List[Dict[str, str]], hits: List[Dict[str, Any]]) -> str:
    history_block = "\n".join([f"- {h.get('role', 'user')}: {h.get('content', '')}" for h in history[-6:]])
    evidence = []
    for i, h in enumerate(hits, start=1):
        p = h["payload"]
        evidence.append(f"[{i}] doc={p.get('doc_id')} domain={p.get('domain_name')} type={p.get('source_spec')}\n{p.get('text', '')}")

    evidence_block = "\n\n".join(evidence)
    return f"""당신은 중·고등학생 진로탐색을 돕는 AI 상담 챗봇입니다.
규칙:
1) 반드시 제공된 근거 안에서만 답하고, 과장하지 마세요.
2) 학생의 흥미/강점/상담맥락을 반영해 '다음 행동'을 제안하세요.
3) 답변 형식: 핵심요약 3줄 + 추천 진로/직업군 + 실천계획(이번주/이번달).
4) 마지막 줄에 [근거] [1],[2] 형태로 인용하세요.

[대화이력]
{history_block if history_block else '(없음)'}

[질문]
{question}

[근거]
{evidence_block}
"""


def call_ollama(prompt: str) -> str:
    url = OLLAMA_BASE_URL.rstrip("/") + "/api/generate"
    response = requests.post(url, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def call_openai(prompt: str) -> str:
    import openai  # type: ignore

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    openai.api_key = OPENAI_API_KEY

    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a student career counseling assistant. Use only provided evidence."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def fallback_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    lines = ["(LLM 미설정) 데이터 기반 상담 참고 내용을 제공합니다.", f"질문: {question}", ""]
    lines.append("핵심 근거:")
    for i, h in enumerate(hits, start=1):
        text = re.sub(r"\s+", " ", h["payload"].get("text", "")).strip()
        lines.append(f"- [{i}] {text[:200]}{'...' if len(text) > 200 else ''}")
    lines.append("\n추천: 위 근거를 바탕으로 학생의 흥미/강점을 확인하고 직업체험 활동 1개를 이번 주에 실행하세요.")
    lines.append("[근거] " + ", ".join([f"[{i}]" for i in range(1, len(hits) + 1)]))
    return "\n".join(lines)


def run_chat(message: str, domain: Optional[str], top_k: int, history: List[Dict[str, str]]) -> ChatResponse:
    qvec = embed_query(message)
    used = "career_all"
    if domain:
        candidate = normalize_collection_name(domain)
        if collection_exists(candidate):
            used = candidate

    hits = search_chunks(used, qvec, top_k)
    citations = []
    for h in hits:
        p = h["payload"]
        excerpt = re.sub(r"\s+", " ", (p.get("text") or "").strip())
        citations.append(
            Citation(
                doc_id=str(p.get("doc_id") or "unknown"),
                domain_name=str(p.get("domain_name") or "unknown"),
                source_spec=p.get("source_spec"),
                excerpt=excerpt[:420] + ("..." if len(excerpt) > 420 else ""),
            )
        )

    if not hits:
        return ChatResponse(answer="관련 근거를 찾지 못했습니다. 더 구체적인 학생 상황을 알려주세요.", citations=[], used_collection=used)

    prompt = build_prompt(message, history, hits)
    if LLM_PROVIDER == "ollama":
        answer = call_ollama(prompt)
    elif LLM_PROVIDER == "openai":
        answer = call_openai(prompt)
    else:
        answer = fallback_answer(message, hits)

    return ChatResponse(answer=answer, citations=citations, used_collection=used)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return run_chat(req.message, req.domain, req.top_k, req.history)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    res = run_chat(req.query, req.domain, req.top_k, [])
    return AskResponse(answer=res.answer, citations=res.citations, used_collection=res.used_collection)


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "qdrant": QDRANT_URL,
        "embed_model": EMBED_MODEL,
        "llm_provider": LLM_PROVIDER,
        "service": "student-career-counsel-chatbot",
    }


app.mount("/", StaticFiles(directory="web", html=True), name="web")
