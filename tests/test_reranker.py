from types import SimpleNamespace
import unittest

import numpy as np

from src.reranking.reranker import CrossEncoderReranker, RerankedHybridRetriever
from src.retrieval.hybrid_retriever import HybridRetrievedChunk


def candidate(rank: int, chunk_id: str, context: str) -> HybridRetrievedChunk:
    return HybridRetrievedChunk(
        rank=rank,
        hybrid_score=1 / (60 + rank),
        dense_rank=rank,
        bm25_rank=None,
        retrieved_by="Dense only",
        chunk_id=chunk_id,
        text=f"Original {chunk_id}",
        document_name="Guideline",
        section_title="Recommendations",
        page_start=1,
        page_end=1,
        contextualized_text=context,
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=np.float32)
        self.calls: list[tuple[list[tuple[str, str]], dict]] = []

    def predict(self, inputs: list[tuple[str, str]], **kwargs):
        self.calls.append((inputs, kwargs))
        return self.scores


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_reranks_contextualized_text_and_preserves_debug_metadata(self) -> None:
        model = FakeCrossEncoder([0.1, 0.9, 0.5])
        reranker = CrossEncoderReranker(model=model, batch_size=2)
        candidates = [
            candidate(1, "general", "general recommendation"),
            candidate(2, "answer", "complete laboratory test list"),
            candidate(3, "background", "testing background"),
        ]

        results = reranker.rerank("Which tests?", candidates, top_k=2)

        self.assertEqual([result.chunk_id for result in results], ["answer", "background"])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual(results[0].original_rank, 2)
        self.assertAlmostEqual(results[0].rerank_score, 0.9)
        self.assertEqual(model.calls[0][0][1], ("Which tests?", "complete laboratory test list"))
        self.assertEqual(model.calls[0][1]["batch_size"], 2)

    def test_pipeline_reranks_the_full_hybrid_candidate_union(self) -> None:
        candidates = [
            candidate(1, "a", "alpha"),
            candidate(2, "b", "beta"),
            candidate(3, "c", "gamma"),
        ]
        hybrid = SimpleNamespace(
            retrieve_candidates=lambda question, candidate_k: candidates,
        )
        reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.1, 0.2, 0.9]))
        pipeline = RerankedHybridRetriever(
            hybrid_retriever=hybrid,
            reranker=reranker,
        )

        results = pipeline.retrieve("question", top_k=2, candidate_k=3)

        self.assertEqual([result.chunk_id for result in results], ["c", "b"])

    def test_rejects_duplicate_candidates(self) -> None:
        reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.1, 0.2]))

        with self.assertRaisesRegex(ValueError, "unique chunk IDs"):
            reranker.rerank(
                "question",
                [candidate(1, "same", "one"), candidate(2, "same", "two")],
            )


if __name__ == "__main__":
    unittest.main()
