"""Interactive entry point for the hybrid Top-K retrieval baseline."""

from __future__ import annotations

import argparse

from src.retrieval import HybridRetrievedChunk, HybridRetriever
from src.retrieval.hybrid_retriever import DEFAULT_CANDIDATE_K


TOP_K = 5
CANDIDATE_K = DEFAULT_CANDIDATE_K
DIVIDER = "-" * 40


def _format_pages(result: HybridRetrievedChunk) -> str:
    if result.page_start is None and result.page_end is None:
        return "Unknown"
    start = result.page_start if result.page_start is not None else "?"
    end = result.page_end if result.page_end is not None else "?"
    return f"{start} - {end}"


def print_results(results: list[HybridRetrievedChunk]) -> None:
    """Print ranked evidence chunks in a human-readable format."""

    if not results:
        print("No relevant chunks were found.")
        return

    for result in results:
        print(f"Rank: {result.rank}")
        print(f"Hybrid Score: {result.hybrid_score:.6f}")
        print(f"Dense Rank: {result.dense_rank or '-'}")
        print(f"BM25 Rank: {result.bm25_rank or '-'}")
        print(f"Retrieved By: {result.retrieved_by}\n")
        print("Chunk ID:")
        print(f"{result.chunk_id}\n")
        print("Text:")
        print(f"{result.text}\n")
        print("Source:")
        print(f"{result.document_name or 'Unknown'}\n")
        print("Section:")
        print(f"{result.section_title or 'Unknown'}\n")
        print("Pages:")
        print(_format_pages(result))
        print(f"\n{DIVIDER}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=CANDIDATE_K,
        help=(
            "Candidates retrieved by each dense/BM25 branch before fusion "
            f"(default: {CANDIDATE_K})"
        ),
    )
    args = parser.parse_args()

    question = input("Enter your question:\n").strip()
    with HybridRetriever() as retriever:
        results = retriever.retrieve(
            question,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
    print_results(results)


if __name__ == "__main__":
    main()
