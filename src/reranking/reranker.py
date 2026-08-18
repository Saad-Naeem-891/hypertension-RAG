"""Cross-encoder reranking over the hybrid Dense/BM25 candidate union."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from sentence_transformers import CrossEncoder

from src.embedding.embed_chunks import DEFAULT_MODEL_CACHE_DIRECTORY
from src.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_K,
    HybridRetrievedChunk,
    HybridRetriever,
)


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_BATCH_SIZE = 16
DEFAULT_COHERE_RERANKER_MODEL = "rerank-v4.0-fast"
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


class CrossEncoderModel(Protocol):
    """Minimal interface needed from a cross-encoder model."""

    def predict(self, inputs: list[tuple[str, str]], **kwargs: Any) -> Any: ...


class ChunkReranker(Protocol):
    """Rerank hybrid candidates for a question."""

    def rerank(
        self,
        query: str,
        candidates: Sequence["HybridRetrievedChunk"],
        *,
        top_k: int = 5,
    ) -> list["RerankedChunk"]: ...


@dataclass(frozen=True, slots=True)
class RerankedChunk:
    """One hybrid candidate after cross-encoder reranking."""

    rank: int
    rerank_score: float
    original_rank: int
    hybrid_score: float
    dense_rank: int | None
    bm25_rank: int | None
    retrieved_by: str
    chunk_id: str
    text: str
    document_name: str | None
    section_title: str | None
    page_start: int | None
    page_end: int | None

    @property
    def rerank(self) -> int:
        """Compatibility alias used by the original reference implementation."""

        return self.rank

    @classmethod
    def from_hybrid(
        cls,
        chunk: HybridRetrievedChunk,
        *,
        rank: int,
        rerank_score: float,
    ) -> "RerankedChunk":
        return cls(
            rank=rank,
            rerank_score=rerank_score,
            original_rank=chunk.rank,
            hybrid_score=chunk.hybrid_score,
            dense_rank=chunk.dense_rank,
            bm25_rank=chunk.bm25_rank,
            retrieved_by=chunk.retrieved_by,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            document_name=chunk.document_name,
            section_title=chunk.section_title,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )


class CrossEncoderReranker:
    """Score question/chunk pairs jointly and return the highest-ranked chunks."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        model_cache_directory: str | Path = DEFAULT_MODEL_CACHE_DIRECTORY,
        device: str = "cpu",
        batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
        model: CrossEncoderModel | None = None,
        local_files_only: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = model or CrossEncoder(
            model_name,
            cache_folder=str(Path(model_cache_directory).expanduser().resolve()),
            device=device,
            local_files_only=local_files_only,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridRetrievedChunk],
        *,
        top_k: int = 5,
    ) -> list[RerankedChunk]:
        """Rerank unique hybrid candidates and return the final Top-K evidence."""

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not candidates:
            return []

        chunk_ids = [candidate.chunk_id for candidate in candidates]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Reranker candidates must have unique chunk IDs")

        pairs = [
            (
                cleaned_query,
                candidate.contextualized_text or candidate.text,
            )
            for candidate in candidates
        ]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError(
                "Cross-encoder returned an unexpected number of reranking scores"
            )

        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].rank, item[0].chunk_id),
        )
        return [
            RerankedChunk.from_hybrid(
                candidate,
                rank=rank,
                rerank_score=float(score),
            )
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        ]


class CohereReranker:
    """Rerank candidates through Cohere's hosted Rerank v2 API."""

    def __init__(
        self,
        model_name: str = DEFAULT_COHERE_RERANKER_MODEL,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise ValueError("COHERE_API_KEY must be set to use the Cohere reranker")

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridRetrievedChunk],
        *,
        top_k: int = 5,
    ) -> list[RerankedChunk]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not candidates:
            return []

        documents = [candidate.contextualized_text or candidate.text for candidate in candidates]
        payload = json.dumps(
            {
                "model": self.model_name,
                "query": cleaned_query,
                "documents": documents,
                "top_n": len(documents),
            }
        ).encode("utf-8")
        request = Request(
            COHERE_RERANK_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(f"Cohere rerank request failed with HTTP {error.code}") from error
        except URLError as error:
            raise RuntimeError("Cohere rerank request could not be completed") from error

        raw_results = body.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != len(candidates):
            raise RuntimeError("Cohere returned an unexpected number of reranking results")

        scores: list[float | None] = [None] * len(candidates)
        for result in raw_results:
            if not isinstance(result, dict):
                raise RuntimeError("Cohere returned an invalid reranking result")
            index = result.get("index")
            score = result.get("relevance_score")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                raise RuntimeError("Cohere returned an invalid candidate index")
            if scores[index] is not None or not isinstance(score, (int, float)):
                raise RuntimeError("Cohere returned invalid reranking scores")
            scores[index] = float(score)
        if any(score is None for score in scores):
            raise RuntimeError("Cohere omitted reranking scores")

        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].rank, item[0].chunk_id),
        )
        return [
            RerankedChunk.from_hybrid(
                candidate,
                rank=rank,
                rerank_score=float(score),
            )
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        ]


class RerankedHybridRetriever:
    """Retrieve a broad hybrid candidate union, then cross-encoder rerank it."""

    def __init__(
        self,
        *,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: ChunkReranker | None = None,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        reranker_batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
        device: str = "cpu",
    ) -> None:
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker(
            reranker_model,
            device=device,
            batch_size=reranker_batch_size,
        )
        self._owns_hybrid_retriever = hybrid_retriever is None

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> list[RerankedChunk]:
        """Return final Top-K chunks after reranking the full candidate union."""

        candidate_limit = max(top_k, candidate_k)
        candidates = self.hybrid_retriever.retrieve_candidates(
            question,
            candidate_k=candidate_limit,
        )
        return self.reranker.rerank(question, candidates, top_k=top_k)

    def close(self) -> None:
        if self._owns_hybrid_retriever:
            self.hybrid_retriever.close()

    def __enter__(self) -> "RerankedHybridRetriever":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
