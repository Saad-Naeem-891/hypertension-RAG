"""Hybrid retrieval using dense Qdrant search, BM25, and rank fusion."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Protocol, Sequence

from src.embedding.embed_chunks import DEFAULT_CHUNKS_DIRECTORY, load_chunks
from src.retrieval.semantic_retriever import RetrievedChunk, SemanticRetriever


DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class BM25Result:
    """One ranked result from the lexical retrieval branch."""

    rank: int
    score: float
    chunk_id: str


@dataclass(frozen=True, slots=True)
class HybridRetrievedChunk:
    """One evidence chunk in the final rank-fused result list."""

    rank: int
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


class DenseRetriever(Protocol):
    """Interface required from the dense retrieval branch."""

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievedChunk]: ...


class LexicalRetriever(Protocol):
    """Interface required from the BM25 retrieval branch."""

    def retrieve(self, question: str, *, top_k: int) -> list[BM25Result]: ...


def tokenize(text: str) -> list[str]:
    """Apply the simple, deterministic tokenizer used by the BM25 baseline."""

    return TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """In-memory BM25 index built from the persisted chunk JSON artifacts."""

    def __init__(
        self,
        chunks: Sequence[dict[str, Any]] | None = None,
        *,
        chunks_directory: str | Path = DEFAULT_CHUNKS_DIRECTORY,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.chunks = list(chunks) if chunks is not None else load_chunks(chunks_directory)
        if not self.chunks:
            raise ValueError("At least one chunk is required to build the BM25 index")

        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self.document_lengths: list[int] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()

        for chunk in self.chunks:
            chunk_id = chunk.get("chunk_id")
            retrieval_text = chunk.get("contextualized_text")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Every BM25 chunk must have a non-empty chunk_id")
            if not isinstance(retrieval_text, str) or not retrieval_text.strip():
                raise ValueError(f"Missing contextualized_text for: {chunk_id}")

            frequencies = Counter(tokenize(retrieval_text))
            self.chunk_ids.append(chunk_id)
            self.document_lengths.append(sum(frequencies.values()))
            self.term_frequencies.append(frequencies)
            self.document_frequencies.update(frequencies.keys())

        self.document_count = len(self.chunks)
        self.average_document_length = (
            sum(self.document_lengths) / self.document_count
        )

    def retrieve(self, question: str, *, top_k: int = DEFAULT_CANDIDATE_K) -> list[BM25Result]:
        """Return the highest-scoring lexical candidates for a question."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_terms = list(dict.fromkeys(tokenize(question.strip())))
        if not query_terms:
            raise ValueError("Question cannot be empty")

        scores: defaultdict[int, float] = defaultdict(float)
        for term in query_terms:
            document_frequency = self.document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue
            inverse_document_frequency = math.log(
                1
                + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

            for index, frequencies in enumerate(self.term_frequencies):
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                length_ratio = (
                    self.document_lengths[index] / self.average_document_length
                    if self.average_document_length
                    else 0.0
                )
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * length_ratio
                )
                scores[index] += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1) / denominator
                )

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.chunk_ids[item[0]]),
        )[:top_k]
        return [
            BM25Result(
                rank=rank,
                score=float(score),
                chunk_id=self.chunk_ids[index],
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]


def _chunk_metadata(chunk: dict[str, Any]) -> tuple[
    str | None,
    str | None,
    int | None,
    int | None,
]:
    metadata = chunk.get("metadata") or {}
    source_file = chunk.get("source_file")
    document_name = metadata.get("document_name")
    if not document_name and isinstance(source_file, str):
        document_name = Path(source_file).stem

    section_title = metadata.get("section_title")
    if section_title is None:
        headings = metadata.get("headings") or []
        section_title = headings[-1] if headings else None

    return (
        document_name,
        section_title,
        metadata.get("page_start"),
        metadata.get("page_end"),
    )


class HybridRetriever:
    """Combine dense and BM25 rankings with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        *,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: LexicalRetriever | None = None,
        chunks: Sequence[dict[str, Any]] | None = None,
        chunks_directory: str | Path = DEFAULT_CHUNKS_DIRECTORY,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")

        loaded_chunks = (
            list(chunks) if chunks is not None else load_chunks(chunks_directory)
        )
        self.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in loaded_chunks}
        if len(self.chunks_by_id) != len(loaded_chunks):
            raise ValueError("Chunk IDs must be unique")

        self.dense_retriever = dense_retriever or SemanticRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever(loaded_chunks)
        self.rrf_k = rrf_k
        self._owns_dense_retriever = dense_retriever is None

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> list[HybridRetrievedChunk]:
        """Retrieve candidates from both branches and return the Top-K RRF ranking."""

        if not question.strip():
            raise ValueError("Question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")

        candidate_limit = max(top_k, candidate_k)
        dense_results = self.dense_retriever.retrieve(
            question,
            top_k=candidate_limit,
        )
        bm25_results = self.bm25_retriever.retrieve(
            question,
            top_k=candidate_limit,
        )

        dense_ranks = {result.chunk_id: result.rank for result in dense_results}
        bm25_ranks = {result.chunk_id: result.rank for result in bm25_results}
        candidate_ids = dense_ranks.keys() | bm25_ranks.keys()

        fused: list[tuple[str, float, int | None, int | None]] = []
        for chunk_id in candidate_ids:
            if chunk_id not in self.chunks_by_id:
                raise ValueError(
                    f"Retrieved chunk is missing from chunk artifacts: {chunk_id}"
                )
            dense_rank = dense_ranks.get(chunk_id)
            bm25_rank = bm25_ranks.get(chunk_id)
            hybrid_score = 0.0
            if dense_rank is not None:
                hybrid_score += 1 / (self.rrf_k + dense_rank)
            if bm25_rank is not None:
                hybrid_score += 1 / (self.rrf_k + bm25_rank)
            fused.append((chunk_id, hybrid_score, dense_rank, bm25_rank))

        fused.sort(
            key=lambda item: (
                -item[1],
                min(rank for rank in item[2:] if rank is not None),
                item[0],
            )
        )

        results: list[HybridRetrievedChunk] = []
        for rank, (chunk_id, score, dense_rank, bm25_rank) in enumerate(
            fused[:top_k],
            start=1,
        ):
            chunk = self.chunks_by_id[chunk_id]
            document_name, section_title, page_start, page_end = _chunk_metadata(chunk)
            if dense_rank is not None and bm25_rank is not None:
                retrieved_by = "Both"
            elif dense_rank is not None:
                retrieved_by = "Dense only"
            else:
                retrieved_by = "BM25 only"

            results.append(
                HybridRetrievedChunk(
                    rank=rank,
                    hybrid_score=score,
                    dense_rank=dense_rank,
                    bm25_rank=bm25_rank,
                    retrieved_by=retrieved_by,
                    chunk_id=chunk_id,
                    text=str(chunk.get("text", "")),
                    document_name=document_name,
                    section_title=section_title,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
        return results

    def close(self) -> None:
        """Close the internally created dense retriever and its Qdrant client."""

        if self._owns_dense_retriever:
            close = getattr(self.dense_retriever, "close", None)
            if close is not None:
                close()

    def __enter__(self) -> HybridRetriever:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
