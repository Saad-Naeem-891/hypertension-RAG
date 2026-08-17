from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
from qdrant_client import QdrantClient

from src.vector_store.qdrant_store import (
    DEFAULT_COLLECTION_NAME,
    index_embeddings,
    point_id_for_chunk,
)


class QdrantStoreTests(unittest.TestCase):
    def test_point_id_is_stable_and_unique(self) -> None:
        self.assertEqual(point_id_for_chunk("chunk_1"), point_id_for_chunk("chunk_1"))
        self.assertNotEqual(point_id_for_chunk("chunk_1"), point_id_for_chunk("chunk_2"))

    def test_index_persists_vectors_and_payloads(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks_directory = root / "chunks"
            chunks_directory.mkdir()
            chunks = [
                {
                    "chunk_id": "chunk_0000",
                    "source_file": "guideline.pdf",
                    "text": "Sodium guidance",
                    "contextualized_text": "Recommendations\nSodium guidance",
                    "metadata": {"headings": ["Recommendations"]},
                },
                {
                    "chunk_id": "chunk_0001",
                    "source_file": "guideline.pdf",
                    "text": "Potassium guidance",
                    "contextualized_text": "Recommendations\nPotassium guidance",
                    "metadata": {"headings": ["Recommendations"]},
                },
            ]
            (chunks_directory / "guideline.json").write_text(
                json.dumps(chunks), encoding="utf-8"
            )
            embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            embeddings_path = root / "embeddings.npy"
            np.save(embeddings_path, embeddings, allow_pickle=False)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "embedding_dimension": 2,
                        "chunks": [
                            {"row_index": 0, "chunk_id": "chunk_0000"},
                            {"row_index": 1, "chunk_id": "chunk_0001"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            database_path = root / "qdrant_db"

            result = index_embeddings(
                database_path,
                embeddings_path=embeddings_path,
                manifest_path=manifest_path,
                chunks_directory=chunks_directory,
            )

            self.assertEqual(result.stored_points, 2)
            client = QdrantClient(path=str(database_path))
            try:
                points = client.retrieve(
                    collection_name=DEFAULT_COLLECTION_NAME,
                    ids=[point_id_for_chunk("chunk_0000")],
                    with_payload=True,
                )
                self.assertEqual(points[0].payload["chunk_id"], "chunk_0000")
                self.assertEqual(points[0].payload["text"], "Sodium guidance")
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
