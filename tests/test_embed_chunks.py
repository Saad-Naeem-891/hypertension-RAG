from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import Mock

import numpy as np

from src.embedding.embed_chunks import (
    DOCUMENT_PREFIX,
    embed_chunks,
    load_chunks,
    save_embedding_artifacts,
)


class EmbedChunksTests(unittest.TestCase):
    def test_load_chunks_is_recursive_and_deterministic(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "WHO").mkdir()
            (root / "WHO" / "b.json").write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": "b_chunk_0000",
                            "contextualized_text": "B evidence",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (root / "a.json").write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": "a_chunk_0000",
                            "contextualized_text": "A evidence",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            chunks = load_chunks(root)

            self.assertEqual([chunk["chunk_id"] for chunk in chunks], ["a_chunk_0000", "b_chunk_0000"])

    def test_embed_chunks_uses_prefix_and_normalization(self) -> None:
        chunks = [{"chunk_id": "chunk_0000", "contextualized_text": "Evidence"}]
        model = Mock()
        model.encode.return_value = np.array([[0.5, 0.5]], dtype=np.float32)

        matrix = embed_chunks(chunks, model, batch_size=2)

        self.assertEqual(matrix.shape, (1, 2))
        model.encode.assert_called_once_with(
            [f"{DOCUMENT_PREFIX}Evidence"],
            batch_size=2,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def test_save_embedding_artifacts_preserves_row_mapping(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            chunks = [
                {
                    "chunk_id": "chunk_0000",
                    "source_file": "guideline.pdf",
                    "contextualized_text": "Evidence",
                }
            ]
            embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

            matrix_path, manifest_path = save_embedding_artifacts(
                embeddings,
                chunks,
                temporary_directory,
                model_name="test-model",
            )

            np.testing.assert_array_equal(np.load(matrix_path), embeddings)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["chunks"][0]["chunk_id"], "chunk_0000")
            self.assertEqual(manifest["embedding_dimension"], 2)


if __name__ == "__main__":
    unittest.main()
