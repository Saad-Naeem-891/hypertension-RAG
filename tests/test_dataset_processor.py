from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.ingestion.dataset_processor import (
    chunk_document,
    discover_pdfs,
    process_dataset,
)


class DatasetProcessorTests(unittest.TestCase):
    def test_discover_pdfs_is_recursive_case_insensitive_and_sorted(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "WHO").mkdir()
            (root / "WHO" / "z.PDF").touch()
            (root / "a.pdf").touch()
            (root / "notes.txt").touch()

            discovered = discover_pdfs(root)

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in discovered],
                ["a.pdf", "WHO/z.PDF"],
            )

    def test_chunk_document_preserves_text_context_and_metadata(self) -> None:
        chunks = [
            SimpleNamespace(
                text="Potassium recommendation",
                meta={"headings": ["Recommendations"], "page": 12},
            )
        ]
        chunker = Mock()
        chunker.chunk.return_value = iter(chunks)
        chunker.contextualize.return_value = "Recommendations\nPotassium recommendation"

        result = chunk_document(
            Mock(),
            source_file="WHO/potassium.pdf",
            document_id="who_potassium",
            chunker=chunker,
        )

        self.assertEqual(
            result,
            [
                {
                    "chunk_id": "who_potassium_chunk_0000",
                    "source_file": "WHO/potassium.pdf",
                    "chunk_index": 0,
                    "text": "Potassium recommendation",
                    "contextualized_text": "Recommendations\nPotassium recommendation",
                    "metadata": {"headings": ["Recommendations"], "page": 12},
                }
            ],
        )

    @patch("src.ingestion.dataset_processor.parse_pdf")
    def test_process_dataset_continues_after_one_pdf_fails(self, parse_pdf: Mock) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "DataSet"
            artifacts_root = root / "artifacts"
            dataset_root.mkdir()
            (dataset_root / "bad.pdf").touch()
            (dataset_root / "good.pdf").touch()

            document = Mock()
            document.export_to_markdown.return_value = "# Good document"
            parse_pdf.side_effect = [RuntimeError("broken PDF"), document]

            chunk = SimpleNamespace(text="Evidence", meta={"headings": ["Evidence"]})
            chunker = Mock()
            chunker.chunk.return_value = iter([chunk])
            chunker.contextualize.return_value = "Evidence\nEvidence"

            summary = process_dataset(
                dataset_root,
                artifacts_root,
                converter=Mock(),
                chunker=chunker,
            )

            self.assertEqual(summary.discovered, 2)
            self.assertEqual(summary.succeeded, 1)
            self.assertEqual(summary.failed, 1)
            self.assertEqual(summary.total_chunks, 1)
            self.assertTrue((artifacts_root / "parsed_markdown" / "good.md").is_file())
            self.assertTrue((artifacts_root / "chunks" / "good.json").is_file())


if __name__ == "__main__":
    unittest.main()
