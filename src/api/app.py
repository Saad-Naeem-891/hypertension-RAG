"""FastAPI service exposing grounded RAG chat and evaluation summaries."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import asynccontextmanager
import csv
from hmac import compare_digest
import json
import logging
import os
from pathlib import Path
from threading import Lock
import time
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.generation import GeminiConfigurationError, GeminiGenerator
from src.reranking import RerankedHybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_RUNS_PATH = PROJECT_ROOT / "artifacts" / "evaluation" / "evaluation_runs.csv"
DEFAULT_REQUESTS_PER_MINUTE = 20
LOGGER = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env", override=False)


class _RateLimiter:
    """Small single-process limiter that protects hosted generation quota."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            timestamps = self._requests[client_id]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.requests_per_minute:
                return False
            timestamps.append(now)
            return True


def _configured_rate_limit() -> int:
    raw_value = os.getenv("RAG_RATE_LIMIT_PER_MINUTE", str(DEFAULT_REQUESTS_PER_MINUTE))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError("RAG_RATE_LIMIT_PER_MINUTE must be an integer") from exc


RATE_LIMITER = _RateLimiter(_configured_rate_limit())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create model/vector resources once and close them on shutdown."""

    retriever = RerankedHybridRetriever()
    app.state.retriever = retriever
    try:
        yield
    finally:
        retriever.close()


app = FastAPI(title="Hypertension RAG API", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-RAG-API-Token"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("message", mode="before")
    @classmethod
    def validate_message(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("message cannot be empty or whitespace")
        return value


class CitationResponse(BaseModel):
    chunk_id: str
    document_name: str | None
    section_title: str | None
    page_start: int | None
    page_end: int | None


class SupportingEvidenceResponse(BaseModel):
    statement: str
    chunk_ids: list[str]


class ChatResponse(BaseModel):
    answer: str
    supporting_evidence: list[SupportingEvidenceResponse]
    confidence: str
    safety_message: str
    citations: list[CitationResponse]


class EvaluationMetricResponse(BaseModel):
    cutoff: int
    precision: float
    recall: float
    hit_rate: float
    mrr: float
    ndcg: float


class EvaluationRunResponse(BaseModel):
    run_id: str
    finished_at_utc: str
    retriever_type: str
    reranker_enabled: bool
    reranker_model: str | None
    candidate_k: int | None
    question_count: int | None
    chunk_count: int | None
    embedding_dimension: int | None
    ground_truth_name: str | None
    metrics: list[EvaluationMetricResponse]


def _optional_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def _metric_rows(raw_metrics: str | None) -> list[EvaluationMetricResponse]:
    if not raw_metrics:
        return []
    try:
        payload = json.loads(raw_metrics)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []

    metrics: list[EvaluationMetricResponse] = []
    for raw_cutoff, values in payload.items():
        if not isinstance(values, dict):
            continue
        try:
            metrics.append(
                EvaluationMetricResponse(
                    cutoff=int(raw_cutoff),
                    precision=float(values["precision"]),
                    recall=float(values["recall"]),
                    hit_rate=float(values["hit"]),
                    mrr=float(values["reciprocal_rank"]),
                    ndcg=float(values["ndcg"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(metrics, key=lambda metric: metric.cutoff)


def _safe_evaluation_row(row: dict[str, str]) -> EvaluationRunResponse:
    ground_truth_path = row.get("ground_truth_path")
    return EvaluationRunResponse(
        run_id=row.get("run_id", ""),
        finished_at_utc=row.get("finished_at_utc", ""),
        retriever_type=row.get("retriever_type", ""),
        reranker_enabled=_as_bool(row.get("reranker_enabled")),
        reranker_model=row.get("reranker_model") or None,
        candidate_k=_optional_int(row.get("candidate_k")),
        question_count=_optional_int(row.get("question_count")),
        chunk_count=_optional_int(row.get("chunk_count")),
        embedding_dimension=_optional_int(row.get("embedding_dimension")),
        ground_truth_name=Path(ground_truth_path).name if ground_truth_path else None,
        metrics=_metric_rows(row.get("metrics_json")),
    )


def _successful_evaluation_rows() -> list[dict[str, str]]:
    if not EVALUATION_RUNS_PATH.is_file():
        return []
    with EVALUATION_RUNS_PATH.open("r", encoding="utf-8", newline="") as source:
        return [
            row
            for row in csv.DictReader(source)
            if row.get("status") == "success"
        ][-12:]


def _require_chat_access(
    request: Request,
    x_rag_api_token: str | None = Header(default=None),
) -> None:
    expected_token = os.getenv("RAG_API_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Chat API authentication is not configured.",
        )
    if not x_rag_api_token or not compare_digest(x_rag_api_token, expected_token):
        raise HTTPException(status_code=401, detail="Unauthorized chat request.")

    client_id = request.client.host if request.client else "unknown"
    if not RATE_LIMITER.allow(client_id):
        raise HTTPException(
            status_code=429,
            detail="Too many chat requests. Please try again in one minute.",
        )


def _retriever(request: Request) -> RerankedHybridRetriever:
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retrieval service is starting.")
    return retriever


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/evaluations", response_model=list[EvaluationRunResponse])
def evaluations() -> list[EvaluationRunResponse]:
    """Return sanitized successful evaluation summaries for the dashboard."""

    return [_safe_evaluation_row(row) for row in _successful_evaluation_rows()]


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(_require_chat_access)],
)
def chat(
    request: ChatRequest,
    retriever: RerankedHybridRetriever = Depends(_retriever),
) -> ChatResponse:
    try:
        evidence = retriever.retrieve(request.message, top_k=request.top_k)
    except Exception:
        LOGGER.exception("Retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="The retrieval service is temporarily unavailable.",
        ) from None
    if not evidence:
        return ChatResponse(
            answer="No relevant guideline evidence was found for this question.",
            supporting_evidence=[],
            confidence="Insufficient Evidence",
            safety_message="This tool provides guideline evidence, not individualized medical advice.",
            citations=[],
        )

    try:
        answer = GeminiGenerator().generate(request.message, evidence)
    except GeminiConfigurationError:
        LOGGER.exception("Gemini is not configured")
        raise HTTPException(
            status_code=503,
            detail="Answer generation is not configured.",
        ) from None
    except Exception:
        LOGGER.exception("Answer generation failed")
        raise HTTPException(
            status_code=502,
            detail="Answer generation is temporarily unavailable.",
        ) from None

    return ChatResponse(
        answer=answer.recommendation,
        supporting_evidence=[
            SupportingEvidenceResponse(
                statement=item.statement,
                chunk_ids=list(item.chunk_ids),
            )
            for item in answer.supporting_evidence
        ],
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
