"""Export chunk IDs and raw chunk text from JSON artifacts to plain text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_DIRECTORY = PROJECT_ROOT / "artifacts" / "chunks"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "chunks_txt"
DIVIDER = "-" * 40


def export_chunk_file(json_path: Path, chunks_root: Path, output_root: Path) -> Path:
    """Write one text file containing only each chunk's ID and raw text."""

    chunks = json.loads(json_path.read_text(encoding="utf-8"))
    sections = [
        f"Chunk ID: {chunk['chunk_id']}\nText: {chunk['text']}"
        for chunk in chunks
    ]

    relative_output = json_path.relative_to(chunks_root).with_suffix(".txt")
    output_path = output_root / relative_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"\n\n{DIVIDER}\n\n".join(sections) + ("\n" if sections else ""),
        encoding="utf-8",
    )
    return output_path


def export_all_chunk_files(
    chunks_directory: Path = DEFAULT_CHUNKS_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> list[Path]:
    """Recursively export every chunk JSON file using the same relative name."""

    chunks_root = chunks_directory.expanduser().resolve()
    output_root = output_directory.expanduser().resolve()
    json_files = sorted(
        chunks_root.rglob("*.json"),
        key=lambda path: path.relative_to(chunks_root).as_posix().casefold(),
    )
    return [export_chunk_file(path, chunks_root, output_root) for path in json_files]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-directory", type=Path, default=DEFAULT_CHUNKS_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()

    output_paths = export_all_chunk_files(args.chunks_directory, args.output_directory)
    print(f"Created {len(output_paths)} text files:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
