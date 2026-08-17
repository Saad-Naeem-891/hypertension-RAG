"""Persist chunk embeddings and citation payloads in local Qdrant."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from src.embedding.embed_chunks import load_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "artifacts" / "qdrant_db"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "artifacts" / "embeddings" / "embeddings.npy"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "embeddings" / "manifest.json"
DEFAULT_CHUNKS_DIRECTORY = PROJECT_ROOT / "artifacts" / "chunks"
DEFAULT_COLLECTION_NAME = "hypertension_guidelines"
DENSE_VECTOR_NAME = "dense"
POINT_NAMESPACE = uuid.UUID("d33f9d22-76a4-4bb7-af61-691d4b97ef46")


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Summary of one Qdrant indexing run."""

    collection_name: str
    database_path: Path
    indexed_points: int
    stored_points: int
    vector_dimension: int


def point_id_for_chunk(chunk_id: str) -> str:
    """Create a stable Qdrant-compatible UUID from a chunk ID."""

    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def load_indexing_data(
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    chunks_directory: str | Path = DEFAULT_CHUNKS_DIRECTORY,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load and cross-check embeddings, the row manifest, and chunk payloads."""

    matrix = np.load(Path(embeddings_path).expanduser().resolve(), allow_pickle=False)
    manifest = json.loads(
        Path(manifest_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in load_chunks(chunks_directory)}
    manifest_rows = manifest.get("chunks", [])

    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, received shape {matrix.shape}")
    if len(manifest_rows) != matrix.shape[0]:
        raise ValueError(
            f"Manifest has {len(manifest_rows)} rows but embeddings have {matrix.shape[0]}"
        )
    if manifest.get("embedding_dimension") != matrix.shape[1]:
        raise ValueError("Manifest embedding dimension does not match the matrix")

    ordered_chunks: list[dict[str, Any]] = []
    for expected_index, row in enumerate(manifest_rows):
        if row.get("row_index") != expected_index:
            raise ValueError(f"Invalid row_index at manifest position {expected_index}")
        chunk_id = row.get("chunk_id")
        if chunk_id not in chunks_by_id:
            raise ValueError(f"Manifest chunk not found in chunk JSON: {chunk_id}")
        ordered_chunks.append(chunks_by_id[chunk_id])

    return np.asarray(matrix, dtype=np.float32), ordered_chunks


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_dimension: int,
    *,
    recreate: bool = False,
) -> None:
    """Create the dense collection, optionally replacing an existing one."""

    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=vector_dimension,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            },
            on_disk_payload=True,
        )
        return

    collection = client.get_collection(collection_name)
    vectors_config = collection.config.params.vectors
    if not isinstance(vectors_config, dict) or DENSE_VECTOR_NAME not in vectors_config:
        raise ValueError(
            f"Collection {collection_name!r} does not contain vector {DENSE_VECTOR_NAME!r}; "
            "run again with --recreate"
        )
    existing_dimension = vectors_config[DENSE_VECTOR_NAME].size
    if existing_dimension != vector_dimension:
        raise ValueError(
            f"Collection dimension is {existing_dimension}, expected {vector_dimension}; "
            "run again with --recreate"
        )


def _point_batches(
    embeddings: np.ndarray,
    chunks: Sequence[dict[str, Any]],
    batch_size: int,
) -> Sequence[list[models.PointStruct]]:
    """Build bounded Qdrant point batches with complete chunk payloads."""

    batches: list[list[models.PointStruct]] = []
    for start in range(0, len(chunks), batch_size):
        batch: list[models.PointStruct] = []
        stop = min(start + batch_size, len(chunks))
        for index in range(start, stop):
            chunk = chunks[index]
            batch.append(
                models.PointStruct(
                    id=point_id_for_chunk(chunk["chunk_id"]),
                    vector={DENSE_VECTOR_NAME: embeddings[index].tolist()},
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "source_file": chunk.get("source_file"),
                        "text": chunk.get("text"),
                        "contextualized_text": chunk.get("contextualized_text"),
                        "metadata": chunk.get("metadata"),
                    },
                )
            )
        batches.append(batch)
    return batches


def index_embeddings(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    chunks_directory: str | Path = DEFAULT_CHUNKS_DIRECTORY,
    batch_size: int = 64,
    recreate: bool = False,
) -> IndexingResult:
    """Upsert all generated embeddings into a persistent local Qdrant database."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    embeddings, chunks = load_indexing_data(
        embeddings_path,
        manifest_path,
        chunks_directory,
    )
    database = Path(database_path).expanduser().resolve()
    database.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(database))

    try:
        ensure_collection(
            client,
            collection_name,
            embeddings.shape[1],
            recreate=recreate,
        )
        for batch in _point_batches(embeddings, chunks, batch_size):
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )
        stored_points = client.count(collection_name=collection_name, exact=True).count
    finally:
        client.close()

    return IndexingResult(
        collection_name=collection_name,
        database_path=database,
        indexed_points=len(chunks),
        stored_points=stored_points,
        vector_dimension=embeddings.shape[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-path", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--chunks-directory", type=Path, default=DEFAULT_CHUNKS_DIRECTORY)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before indexing",
    )
    args = parser.parse_args()

    result = index_embeddings(
        args.database_path,
        args.collection_name,
        embeddings_path=args.embeddings_path,
        manifest_path=args.manifest_path,
        chunks_directory=args.chunks_directory,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )
    print(f"Database: {result.database_path}")
    print(f"Collection: {result.collection_name}")
    print(f"Vector dimension: {result.vector_dimension}")
    print(f"Points indexed: {result.indexed_points}")
    print(f"Points stored: {result.stored_points}")


if __name__ == "__main__":
    main()
