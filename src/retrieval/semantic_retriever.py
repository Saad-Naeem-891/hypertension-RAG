"""Dense semantic Top-K retrieval over the persistent Qdrant collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.embedding.embed_chunks import DEFAULT_MODEL_CACHE_DIRECTORY
from src.embedding.model_config import QUERY_PREFIX
from src.vector_store.qdrant_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATABASE_PATH,
    DEFAULT_MANIFEST_PATH,
    DENSE_VECTOR_NAME,
)

class QueryEmbeddingModel(Protocol):
    """Minimal interface required to embed a retrieval question."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One ranked evidence chunk returned by semantic retrieval."""

    rank: int
    score: float
    chunk_id: str
    text: str
    document_name: str | None
    section_title: str | None
    page_start: int | None
    page_end: int | None


def _page_range_from_docling(metadata: dict[str, Any]) -> tuple[int | None, int | None]:
    """Recover a page range from raw Docling provenance as a compatibility fallback."""

    page_numbers = [
        provenance["page_no"]
        for item in metadata.get("doc_items") or []
        if isinstance(item, dict)
        for provenance in item.get("prov") or []
        if isinstance(provenance, dict)
        and isinstance(provenance.get("page_no"), int)
    ]
    if not page_numbers:
        return None, None
    return min(page_numbers), max(page_numbers)


def _result_from_point(point: Any, rank: int) -> RetrievedChunk:
    """Convert a Qdrant scored point into the public retrieval result."""

    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    source_file = payload.get("source_file")
    document_name = metadata.get("document_name")
    if not document_name and isinstance(source_file, str):
        document_name = Path(source_file).stem

    section_title = metadata.get("section_title")
    if section_title is None:
        headings = metadata.get("headings") or []
        section_title = headings[-1] if headings else None

    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    if page_start is None or page_end is None:
        fallback_start, fallback_end = _page_range_from_docling(metadata)
        page_start = fallback_start if page_start is None else page_start
        page_end = fallback_end if page_end is None else page_end

    return RetrievedChunk(
        rank=rank,
        score=float(point.score),
        chunk_id=str(payload.get("chunk_id", "")),
        text=str(payload.get("text", "")),
        document_name=document_name,
        section_title=section_title,
        page_start=page_start,
        page_end=page_end,
    )


class SemanticRetriever:
    """Embed questions and retrieve the nearest dense vectors from Qdrant."""

    def __init__(
        self,
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        model_cache_directory: str | Path = DEFAULT_MODEL_CACHE_DIRECTORY,
        device: str = "cpu",
        model: QueryEmbeddingModel | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        manifest = json.loads(
            Path(manifest_path).expanduser().resolve().read_text(encoding="utf-8")
        )
        if manifest.get("normalized") is not True:
            raise ValueError("Document embeddings must be normalized for this retriever")

        self.embedding_dimension = int(manifest["embedding_dimension"])
        self.model_name = str(manifest["model_name"])
        self.query_prefix = str(manifest.get("query_prefix", QUERY_PREFIX))
        self.collection_name = collection_name
        self.model = model or SentenceTransformer(
            self.model_name,
            cache_folder=str(Path(model_cache_directory).expanduser().resolve()),
            device=device,
            local_files_only=True,
        )
        self.client = client or QdrantClient(
            path=str(Path(database_path).expanduser().resolve())
        )
        self._owns_client = client is None

    def embed_question(self, question: str) -> np.ndarray:
        """Embed a question with the model's retrieval instruction and normalization."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty")

        embeddings = self.model.encode(
            [f"{self.query_prefix}{cleaned_question}"],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_vector = np.asarray(embeddings, dtype=np.float32)
        expected_shape = (1, self.embedding_dimension)
        if query_vector.shape != expected_shape:
            raise ValueError(
                f"Unexpected query embedding shape {query_vector.shape}; "
                f"expected {expected_shape}"
            )
        return query_vector[0]

    def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the Top-K nearest evidence chunks for a user question."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = self.embed_question(question)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            _result_from_point(point, rank)
            for rank, point in enumerate(response.points, start=1)
        ]

    def close(self) -> None:
        """Close the locally owned Qdrant connection."""

        if self._owns_client:
            self.client.close()

    def __enter__(self) -> SemanticRetriever:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
