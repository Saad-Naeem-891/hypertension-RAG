"""FastAPI service exposing grounded RAG chat and evaluation summaries."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.generation import GeminiConfigurationError, GeminiGenerator
from src.reranking import RerankedHybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_RUNS_PATH = PROJECT_ROOT / "artifacts" / "evaluation" / "evaluation_runs.csv"

app = FastAPI(title="Hypertension RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)


class CitationResponse(BaseModel):
    chunk_id: str
    document_name: str | None
    section_title: str | None
    page_start: int | None
    page_end: int | None


class ChatResponse(BaseModel):
    answer: str
    confidence: str
    safety_message: str
    citations: list[CitationResponse]


@lru_cache(maxsize=1)
def _retriever() -> RerankedHybridRetriever:
    return RerankedHybridRetriever()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/evaluations")
def evaluations() -> list[dict[str, str]]:
    """Return successful historical evaluations for the dashboard."""

    if not EVALUATION_RUNS_PATH.is_file():
        return []
    with EVALUATION_RUNS_PATH.open("r", encoding="utf-8", newline="") as source:
        return [
            row
            for row in csv.DictReader(source)
            if row.get("status") == "success"
        ][-12:]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.message.strip()
    try:
        evidence = _retriever().retrieve(question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {exc}") from exc
    if not evidence:
        return ChatResponse(
            answer="No relevant guideline evidence was found for this question.",
            confidence="insufficient_evidence",
            safety_message="This tool provides guideline evidence, not individualized medical advice.",
            citations=[],
        )

    try:
        answer = GeminiGenerator().generate(question, evidence)
    except GeminiConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc

    return ChatResponse(
        answer=answer.recommendation,
        confidence=answer.confidence,
        safety_message=answer.safety_message,
        citations=[
            CitationResponse(
                chunk_id=citation.chunk_id,
                document_name=citation.document_name,
                section_title=citation.section_title,
                page_start=citation.page_start,
                page_end=citation.page_end,
            )
            for citation in answer.citations
        ],
    )
