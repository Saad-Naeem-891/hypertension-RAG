"""Interactive entry point for grounded RAG answers using xAI Grok."""

from __future__ import annotations

import argparse
import os

from src.generation import (
    DEFAULT_GROK_MODEL,
    GrokConfigurationError,
    GrokGenerationError,
    GrokGenerator,
)
from src.retrieval.hybrid_retriever import DEFAULT_CANDIDATE_K
from src.reranking import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MODEL,
    RerankedChunk,
    RerankedHybridRetriever,
)


TOP_K = 5
CANDIDATE_K = DEFAULT_CANDIDATE_K
DIVIDER = "-" * 40


def _format_pages(result: RerankedChunk) -> str:
    if result.page_start is None and result.page_end is None:
        return "Unknown"
    start = result.page_start if result.page_start is not None else "?"
    end = result.page_end if result.page_end is not None else "?"
    return f"{start} - {end}"


def print_results(results: list[RerankedChunk]) -> None:
    """Print ranked evidence chunks in a human-readable format."""

    if not results:
        print("No relevant chunks were found.")
        return

    for result in results:
        print(f"Rank: {result.rank}")
        print(f"Reranker Score: {result.rerank_score:.6f}")
        print(f"Original Hybrid Rank: {result.original_rank}")
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


def print_sources(results: list[RerankedChunk]) -> None:
    """Print a compact list of evidence supplied to the generation model."""

    print("Evidence supplied to Grok:")
    for result in results:
        source = result.document_name or "Unknown"
        section = result.section_title or "Unknown"
        pages = _format_pages(result)
        print(
            f"- [{result.chunk_id}] {source}; "
            f"section: {section}; pages: {pages}"
        )


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
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Cross-encoder model (default: {DEFAULT_RERANKER_MODEL})",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=DEFAULT_RERANKER_BATCH_SIZE,
        help=(
            "Question/chunk pairs scored per reranker batch "
            f"(default: {DEFAULT_RERANKER_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--grok-model",
        default=os.getenv("XAI_MODEL", DEFAULT_GROK_MODEL),
        help=(
            "xAI model used for grounded generation "
            f"(default: XAI_MODEL or {DEFAULT_GROK_MODEL})"
        ),
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=60.0,
        help="Grok API timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Print reranked chunks without calling Grok",
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print the complete retrieved chunks after the generated answer",
    )
    args = parser.parse_args()

    generator = None
    if not args.retrieval_only:
        try:
            generator = GrokGenerator(
                model=args.grok_model,
                timeout_seconds=args.api_timeout,
            )
        except GrokConfigurationError as exc:
            parser.error(str(exc))

    question = input("Enter your question:\n").strip()
    with RerankedHybridRetriever(
        reranker_model=args.reranker_model,
        reranker_batch_size=args.reranker_batch_size,
    ) as retriever:
        results = retriever.retrieve(
            question,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )

    if args.retrieval_only:
        print_results(results)
        return
    if not results:
        print("No relevant chunks were found, so no answer was generated.")
        return

    assert generator is not None
    try:
        answer = generator.generate(question, results)
    except GrokGenerationError as exc:
        raise SystemExit(f"Generation failed: {exc}") from exc

    print("\nAnswer:")
    print(answer.text)
    print(f"\nProvider: {answer.provider}")
    print(f"Model: {answer.model}\n")
    print_sources(results)

    if args.show_evidence:
        print(f"\n{DIVIDER}\n")
        print_results(results)


if __name__ == "__main__":
    main()
