"""Generate normalized dense embeddings for Docling chunk artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_DIRECTORY = PROJECT_ROOT / "artifacts" / "chunks"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "embeddings"
DEFAULT_MODEL_CACHE_DIRECTORY = PROJECT_ROOT / "models"
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DOCUMENT_PREFIX = "passage: "


class EmbeddingModel(Protocol):
    """Minimal model interface used by the embedding pipeline."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> np.ndarray: ...


def load_chunks(chunks_directory: str | Path) -> list[dict[str, Any]]:
    """Load all chunk JSON files recursively in deterministic order."""

    root = Path(chunks_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Chunks directory does not exist: {root}")

    json_files = sorted(
        root.rglob("*.json"),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    if not json_files:
        raise FileNotFoundError(f"No chunk JSON files found in: {root}")

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for json_path in json_files:
        records = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array in: {json_path}")

        for record in records:
            chunk_id = record.get("chunk_id")
            text = record.get("contextualized_text")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Missing chunk_id in: {json_path}")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing contextualized_text for: {chunk_id}")

            seen_ids.add(chunk_id)
            chunks.append(record)

    return chunks


def prepare_passages(chunks: Sequence[dict[str, Any]]) -> list[str]:
    """Apply the E5 document prefix to each contextualized chunk."""

    return [f"{DOCUMENT_PREFIX}{chunk['contextualized_text']}" for chunk in chunks]


def embed_chunks(
    chunks: Sequence[dict[str, Any]],
    model: EmbeddingModel,
    *,
    batch_size: int = 4,
) -> np.ndarray:
    """Create normalized float32 embeddings for all chunks."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    embeddings = model.encode(
        prepare_passages(chunks),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError(
            f"Unexpected embedding shape {matrix.shape}; expected {len(chunks)} rows"
        )
    return matrix


def save_embedding_artifacts(
    embeddings: np.ndarray,
    chunks: Sequence[dict[str, Any]],
    output_directory: str | Path,
    *,
    model_name: str,
) -> tuple[Path, Path]:
    """Save the matrix and its row-to-chunk mapping without a vector database."""

    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_root / "embeddings.npy"
    manifest_path = output_root / "manifest.json"

    np.save(embeddings_path, embeddings, allow_pickle=False)
    manifest = {
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "chunk_count": len(chunks),
        "text_field": "contextualized_text",
        "document_prefix": DOCUMENT_PREFIX,
        "normalized": True,
        "dtype": str(embeddings.dtype),
        "chunks": [
            {
                "row_index": index,
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk.get("source_file"),
            }
            for index, chunk in enumerate(chunks)
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return embeddings_path, manifest_path


def run_embedding_pipeline(
    chunks_directory: str | Path = DEFAULT_CHUNKS_DIRECTORY,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    model_cache_directory: str | Path = DEFAULT_MODEL_CACHE_DIRECTORY,
    batch_size: int = 4,
    device: str = "cpu",
) -> tuple[np.ndarray, Path, Path]:
    """Load chunks, embed them with a local model, and save the artifacts."""

    chunks = load_chunks(chunks_directory)
    model = SentenceTransformer(
        model_name,
        cache_folder=str(Path(model_cache_directory).expanduser().resolve()),
        device=device,
        local_files_only=True,
    )
    embeddings = embed_chunks(chunks, model, batch_size=batch_size)
    embeddings_path, manifest_path = save_embedding_artifacts(
        embeddings,
        chunks,
        output_directory,
        model_name=model_name,
    )
    return embeddings, embeddings_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-directory", type=Path, default=DEFAULT_CHUNKS_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-cache-directory",
        type=Path,
        default=DEFAULT_MODEL_CACHE_DIRECTORY,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks_directory)
    print(f"Chunks loaded: {len(chunks)}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    embeddings, embeddings_path, manifest_path = run_embedding_pipeline(
        args.chunks_directory,
        args.output_directory,
        model_name=args.model_name,
        model_cache_directory=args.model_cache_directory,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Embedding matrix: {embeddings.shape}")
    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
