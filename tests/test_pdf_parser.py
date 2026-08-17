from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.ingestion.pdf_parser import export_to_markdown, parse_pdf


class PdfParserTests(unittest.TestCase):
    def test_parse_pdf_returns_the_converted_document(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "guideline.pdf"
            pdf_path.write_bytes(b"%PDF-1.7\n")
            expected_document = Mock()
            converter = Mock()
            converter.convert.return_value = SimpleNamespace(document=expected_document)

            document = parse_pdf(pdf_path, converter=converter)

            self.assertIs(document, expected_document)
            converter.convert.assert_called_once_with(str(pdf_path.resolve()))

    def test_parse_pdf_rejects_missing_file(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            parse_pdf("missing.pdf", converter=Mock())

    def test_parse_pdf_rejects_non_pdf_file(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            text_path = Path(temporary_directory) / "guideline.txt"
            text_path.write_text("not a PDF", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Expected a PDF"):
                parse_pdf(text_path, converter=Mock())

    def test_export_to_markdown_delegates_to_document(self) -> None:
        document = Mock()
        document.export_to_markdown.return_value = "# Parsed guideline"

        markdown = export_to_markdown(document)

        self.assertEqual(markdown, "# Parsed guideline")
        document.export_to_markdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
