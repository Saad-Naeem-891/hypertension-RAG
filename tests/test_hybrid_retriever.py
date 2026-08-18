from types import SimpleNamespace
import unittest

from src.retrieval.hybrid_retriever import (
    BM25Result,
    BM25Retriever,
    HybridRetriever,
)
from src.retrieval.semantic_retriever import RetrievedChunk


def make_chunk(chunk_id: str, contextualized_text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_file": "guideline.pdf",
        "text": f"Original text for {chunk_id}",
        "contextualized_text": contextualized_text,
        "metadata": {
            "document_name": "Guideline",
            "section_title": "Recommendations",
            "page_start": 3,
            "page_end": 4,
        },
    }


def dense_result(rank: int, chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        score=1.0 / rank,
        chunk_id=chunk_id,
        text="",
        document_name=None,
        section_title=None,
        page_start=None,
        page_end=None,
    )


class BM25RetrieverTests(unittest.TestCase):
    def test_exact_keyword_match_ranks_first(self) -> None:
        chunks = [
            make_chunk("sodium", "sodium intake recommendation adults sodium"),
            make_chunk("potassium", "potassium intake recommendation adults"),
            make_chunk("medicine", "pharmacological treatment hypertension"),
        ]
        retriever = BM25Retriever(chunks)

        results = retriever.retrieve("recommended sodium intake", top_k=2)

        self.assertEqual(results[0].chunk_id, "sodium")
        self.assertGreater(results[0].score, 0)
        self.assertLessEqual(len(results), 2)


class HybridRetrieverTests(unittest.TestCase):
    def test_rrf_merges_duplicates_and_preserves_debug_ranks(self) -> None:
        chunks = [
            make_chunk("a", "alpha"),
            make_chunk("b", "beta"),
            make_chunk("c", "gamma"),
        ]
        dense = SimpleNamespace(
            retrieve=lambda question, top_k: [
                dense_result(1, "a"),
                dense_result(2, "b"),
            ]
        )
        bm25 = SimpleNamespace(
            retrieve=lambda question, top_k: [
                BM25Result(rank=1, score=4.0, chunk_id="c"),
                BM25Result(rank=2, score=3.0, chunk_id="a"),
            ]
        )
        retriever = HybridRetriever(
            chunks=chunks,
            dense_retriever=dense,
            bm25_retriever=bm25,
        )

        results = retriever.retrieve("question", top_k=3, candidate_k=2)

        self.assertEqual([result.chunk_id for result in results], ["a", "c", "b"])
        self.assertEqual(len({result.chunk_id for result in results}), 3)
        self.assertEqual(results[0].dense_rank, 1)
        self.assertEqual(results[0].bm25_rank, 2)
        self.assertEqual(results[0].retrieved_by, "Both")
        self.assertEqual(results[1].retrieved_by, "BM25 only")
        self.assertEqual(results[2].retrieved_by, "Dense only")
        self.assertEqual(results[0].text, "Original text for a")
        self.assertEqual(results[0].document_name, "Guideline")
        self.assertEqual(results[0].page_start, 3)
        self.assertEqual(results[0].page_end, 4)
        self.assertEqual(results[0].contextualized_text, "alpha")

    def test_candidate_retrieval_returns_full_deduplicated_union(self) -> None:
        chunks = [make_chunk(chunk_id, chunk_id) for chunk_id in ("a", "b", "c")]
        dense = SimpleNamespace(
            retrieve=lambda question, top_k: [
                dense_result(1, "a"),
                dense_result(2, "b"),
            ]
        )
        bm25 = SimpleNamespace(
            retrieve=lambda question, top_k: [
                BM25Result(rank=1, score=2.0, chunk_id="a"),
                BM25Result(rank=2, score=1.0, chunk_id="c"),
            ]
        )
        retriever = HybridRetriever(
            chunks=chunks,
            dense_retriever=dense,
            bm25_retriever=bm25,
        )

        results = retriever.retrieve_candidates("question", candidate_k=2)

        self.assertEqual(len(results), 3)
        self.assertEqual({result.chunk_id for result in results}, {"a", "b", "c"})

    def test_returns_exact_top_k_when_candidates_are_available(self) -> None:
        chunks = [make_chunk(str(index), f"term {index}") for index in range(6)]
        dense = SimpleNamespace(
            retrieve=lambda question, top_k: [
                dense_result(rank, str(rank - 1)) for rank in range(1, 7)
            ]
        )
        bm25 = SimpleNamespace(retrieve=lambda question, top_k: [])
        retriever = HybridRetriever(
            chunks=chunks,
            dense_retriever=dense,
            bm25_retriever=bm25,
        )

        results = retriever.retrieve("question", top_k=5, candidate_k=3)

        self.assertEqual(len(results), 5)
        self.assertEqual([result.rank for result in results], [1, 2, 3, 4, 5])

    def test_empty_question_is_rejected(self) -> None:
        chunk = make_chunk("a", "alpha")
        retriever = HybridRetriever(
            chunks=[chunk],
            dense_retriever=SimpleNamespace(retrieve=lambda question, top_k: []),
            bm25_retriever=SimpleNamespace(retrieve=lambda question, top_k: []),
        )

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            retriever.retrieve("   ")


if __name__ == "__main__":
    unittest.main()
