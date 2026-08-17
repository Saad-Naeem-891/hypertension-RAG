"""Parse PDF guideline files into structured Docling documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument


def _validate_pdf_path(pdf_path: str | Path) -> Path:
    """Resolve and validate a local PDF path."""

    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")
    return path


def parse_pdf(
    pdf_path: str | Path,
    *,
    converter: DocumentConverter | None = None,
) -> DoclingDocument:
    """Convert one local PDF into a structured :class:`DoclingDocument`.

    A converter can be supplied when parsing several PDFs so the same converter
    instance and its initialized pipeline can be reused.
    """

    path = _validate_pdf_path(pdf_path)
    document_converter = converter or DocumentConverter()
    result = document_converter.convert(str(path))
    return result.document


def export_to_markdown(document: DoclingDocument) -> str:
    """Export a parsed Docling document to Markdown for manual inspection."""

    return document.export_to_markdown()


def save_markdown(document: DoclingDocument, output_path: str | Path) -> Path:
    """Export a parsed document and save it as UTF-8 Markdown."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(export_to_markdown(document), encoding="utf-8")
    return path


def parse_pdf_to_markdown(
    pdf_path: str | Path,
    *,
    converter: DocumentConverter | None = None,
) -> str:
    """Parse one PDF and return its Markdown representation."""

    document = parse_pdf(pdf_path, converter=converter)
    return export_to_markdown(document)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a PDF with Docling and inspect its Markdown output."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF guideline")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write Markdown to this file instead of printing it",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    markdown = parse_pdf_to_markdown(args.pdf_path)

    if args.output is None:
        print(markdown)
        return

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Markdown written to: {output_path}")


if __name__ == "__main__":
    main()
