"""Parse and chunk every PDF in the guideline dataset with Docling."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

from .pdf_parser import parse_pdf, save_markdown


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIRECTORY = PROJECT_ROOT / "DataSet"
DEFAULT_ARTIFACTS_DIRECTORY = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True, slots=True)
class ProcessingFailure:
    """Information about a PDF that could not be processed."""

    source_file: str
    error: str


@dataclass(frozen=True, slots=True)
class ProcessingSummary:
    """Aggregate results from one dataset-processing run."""

    discovered: int
    succeeded: int
    failed: int
    total_chunks: int
    failures: tuple[ProcessingFailure, ...]


def discover_pdfs(dataset_directory: str | Path) -> list[Path]:
    """Recursively discover PDF files in deterministic relative-path order."""

    root = Path(dataset_directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def _document_id(relative_pdf_path: Path) -> str:
    """Build a stable, filesystem-safe ID from a dataset-relative PDF path."""

    path_without_suffix = relative_pdf_path.with_suffix("")
    raw_id = "__".join(path_without_suffix.parts)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", raw_id).strip("_").lower()
    return normalized or "document"


def to_json_safe(value: Any) -> Any:
    """Convert Docling/Pydantic metadata into JSON-safe Python values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_json_safe(value.value)
    if hasattr(value, "model_dump"):
        return to_json_safe(value.model_dump(mode="json", by_alias=True))
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_safe(item) for item in value]
    return str(value)


def chunk_document(
    document: DoclingDocument,
    *,
    source_file: str,
    document_id: str,
    chunker: HybridChunker | None = None,
) -> list[dict[str, Any]]:
    """Create structured and contextualized chunks from a Docling document."""

    document_chunker = chunker or HybridChunker()
    serialized_chunks: list[dict[str, Any]] = []

    for index, chunk in enumerate(document_chunker.chunk(dl_doc=document)):
        serialized_chunks.append(
            {
                "chunk_id": f"{document_id}_chunk_{index:04d}",
                "source_file": source_file,
                "chunk_index": index,
                "text": chunk.text,
                "contextualized_text": document_chunker.contextualize(chunk=chunk),
                "metadata": to_json_safe(chunk.meta),
            }
        )

    return serialized_chunks


def serialize_chunks(chunks: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Save one document's chunks as formatted UTF-8 JSON."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(chunks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def process_dataset(
    dataset_directory: str | Path = DEFAULT_DATASET_DIRECTORY,
    artifacts_directory: str | Path = DEFAULT_ARTIFACTS_DIRECTORY,
    *,
    converter: DocumentConverter | None = None,
    chunker: HybridChunker | None = None,
) -> ProcessingSummary:
    """Parse, export, and chunk all PDFs while isolating per-file failures."""

    dataset_root = Path(dataset_directory).expanduser().resolve()
    artifacts_root = Path(artifacts_directory).expanduser().resolve()
    pdf_paths = discover_pdfs(dataset_root)
    document_converter = converter or DocumentConverter()
    document_chunker = chunker or HybridChunker()

    succeeded = 0
    total_chunks = 0
    failures: list[ProcessingFailure] = []

    for pdf_path in pdf_paths:
        relative_pdf = pdf_path.relative_to(dataset_root)
        source_file = relative_pdf.as_posix()
        markdown_path = artifacts_root / "parsed_markdown" / relative_pdf.with_suffix(".md")
        chunks_path = artifacts_root / "chunks" / relative_pdf.with_suffix(".json")

        print(f"Processing: {source_file}")
        try:
            document = parse_pdf(pdf_path, converter=document_converter)
            save_markdown(document, markdown_path)
            chunks = chunk_document(
                document,
                source_file=source_file,
                document_id=_document_id(relative_pdf),
                chunker=document_chunker,
            )
            serialize_chunks(chunks, chunks_path)
        except Exception as exc:
            failure = ProcessingFailure(source_file=source_file, error=str(exc))
            failures.append(failure)
            print(f"Failed: {source_file}: {exc}")
            continue

        succeeded += 1
        total_chunks += len(chunks)
        print(f"Completed: {source_file} ({len(chunks)} chunks)")

    return ProcessingSummary(
        discovered=len(pdf_paths),
        succeeded=succeeded,
        failed=len(failures),
        total_chunks=total_chunks,
        failures=tuple(failures),
    )


def print_summary(summary: ProcessingSummary) -> None:
    """Print a compact processing summary and any file-level failures."""

    print("\nProcessing summary")
    print(f"PDFs discovered: {summary.discovered}")
    print(f"Successfully processed: {summary.succeeded}")
    print(f"Failed: {summary.failed}")
    print(f"Total chunks created: {summary.total_chunks}")

    if summary.failures:
        print("\nFailures:")
        for failure in summary.failures:
            print(f"- {failure.source_file}: {failure.error}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse and chunk every PDF in a dataset with Docling."
    )
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=DEFAULT_DATASET_DIRECTORY,
        help=f"Dataset root (default: {DEFAULT_DATASET_DIRECTORY})",
    )
    parser.add_argument(
        "--artifacts-directory",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIRECTORY,
        help=f"Output root (default: {DEFAULT_ARTIFACTS_DIRECTORY})",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = process_dataset(args.dataset_directory, args.artifacts_directory)
    print_summary(summary)


if __name__ == "__main__":
    main()
