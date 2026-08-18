from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock

import numpy as np

from src.retrieval.semantic_retriever import QUERY_PREFIX, SemanticRetriever


class SemanticRetrieverTests(unittest.TestCase):
    def _manifest(self, directory: str) -> Path:
        path = Path(directory) / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "model_name": "test-model",
                    "embedding_dimension": 2,
                    "normalized": True,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_retrieve_embeds_query_and_returns_top_k_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            model = Mock()
            model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
            points = [
                SimpleNamespace(
                    score=0.9 - index * 0.1,
                    payload={
                        "chunk_id": f"chunk_{index:04d}",
                        "text": f"Evidence {index}",
                        "source_file": "guideline.pdf",
                        "metadata": {
                            "document_name": "Guideline",
                            "section_title": "Recommendations",
                            "page_start": index + 1,
                            "page_end": index + 1,
                        },
                    },
                )
                for index in range(3)
            ]
            client = Mock()
            client.query_points.return_value = SimpleNamespace(points=points)
            retriever = SemanticRetriever(
                manifest_path=self._manifest(temporary_directory),
                model=model,
                client=client,
            )

            results = retriever.retrieve("sodium intake", top_k=3)

            self.assertEqual(len(results), 3)
            self.assertEqual(results[0].chunk_id, "chunk_0000")
            self.assertEqual(results[0].document_name, "Guideline")
            self.assertEqual(results[0].section_title, "Recommendations")
            self.assertEqual(results[0].page_start, 1)
            model.encode.assert_called_once_with(
                [f"{QUERY_PREFIX}sodium intake"],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            client.query_points.assert_called_once_with(
                collection_name="hypertension_guidelines",
                query=[1.0, 0.0],
                using="dense",
                limit=3,
                with_payload=True,
                with_vectors=False,
            )

    def test_empty_question_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            retriever = SemanticRetriever(
                manifest_path=self._manifest(temporary_directory),
                model=Mock(),
                client=Mock(),
            )

            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                retriever.retrieve("   ")


if __name__ == "__main__":
    unittest.main()
