"""Manual end-to-end smoke check that retrieves evidence and calls Gemini."""

from __future__ import annotations

from src.generation import GeminiGenerator
from src.reranking import RerankedHybridRetriever


def main() -> None:
    question = "What is the recommended sodium intake?"
    with RerankedHybridRetriever() as retriever:
        evidence = retriever.retrieve(question, top_k=3)
        if not evidence:
            raise RuntimeError("No evidence was retrieved")

        for chunk in evidence:
            print(
                f"Chunk {chunk.rank}: {chunk.chunk_id} "
                f"(score={chunk.rerank_score:.4f})"
            )

        answer = GeminiGenerator().generate(question, evidence)
        print("\nRecommendation:\n" + answer.recommendation)


if __name__ == "__main__":
    main()
