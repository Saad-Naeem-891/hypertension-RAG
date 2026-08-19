"""Interactive entry point for grounded RAG answers using a hosted model."""

from __future__ import annotations

import argparse
import os

from src.generation import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROK_MODEL,
    Citation,
    GeneratedAnswer,
    GeminiConfigurationError,
    GeminiGenerationError,
    GeminiGenerator,
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
SECTION_DIVIDER = "=" * 50


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

    print("Evidence supplied to the generation model:")
    for index, result in enumerate(results, start=1):
        print(f"{index}. {result.chunk_id}")


def _format_citation_pages(answer_citation: Citation) -> str:
    page_start = answer_citation.page_start
    page_end = answer_citation.page_end
    if page_start is None and page_end is None:
        return "Unknown"
    start = page_start if page_start is not None else "?"
    end = page_end if page_end is not None else "?"
    return str(start) if start == end else f"{start}-{end}"


def print_final_response(answer: GeneratedAnswer) -> None:
    """Render the validated grounded answer for the end user."""

    citation_numbers = {
        citation.chunk_id: index
        for index, citation in enumerate(answer.citations, start=1)
    }

    print(f"\n{SECTION_DIVIDER}")
    print("FINAL RESPONSE")
    print(SECTION_DIVIDER)
    print("\nRecommendation:")
    print(answer.recommendation)

    print("\nSupporting Evidence:")
    if answer.supporting_evidence:
        for item in answer.supporting_evidence:
            references = ", ".join(
                str(citation_numbers[chunk_id]) for chunk_id in item.chunk_ids
            )
            print(f"- {item.statement} [{references}]")
    else:
        print("- No sufficient supporting evidence was found.")

    print("\nCitations:")
    if answer.citations:
        for index, citation in enumerate(answer.citations, start=1):
            print(f"\n[{index}]")
            print(f"Document: {citation.document_name or 'Unknown'}")
            print(f"Section: {citation.section_title or 'Unknown'}")
            print(f"Pages: {_format_citation_pages(citation)}")
            print(f"Chunk ID: {citation.chunk_id}")
    else:
        print("None.")

    print("\nConfidence:")
    print(answer.confidence)
    print("\nSafety:")
    print(answer.safety_message)


def print_debug_information(
    answer: GeneratedAnswer,
    results: list[RerankedChunk],
) -> None:
    """Print provider and retrieval details separately from the final response."""

    print(f"\n{SECTION_DIVIDER}")
    print("DEBUG / RETRIEVAL INFORMATION")
    print(SECTION_DIVIDER)
    print("\nProvider:")
    print(answer.provider)
    print("\nModel:")
    print(answer.model)
    print()
    print_sources(results)


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
        "--provider",
        choices=("gemini", "grok"),
        default=os.getenv("GENERATION_PROVIDER", "gemini"),
        help="Hosted generation provider (default: GENERATION_PROVIDER or gemini)",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        help=(
            "Gemini model used for grounded generation "
            f"(default: GEMINI_MODEL or {DEFAULT_GEMINI_MODEL})"
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
        help="Hosted API timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Print reranked chunks without calling a hosted model",
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
            if args.provider == "gemini":
                generator = GeminiGenerator(
                    model=args.gemini_model,
                    timeout_seconds=args.api_timeout,
                )
            else:
                generator = GrokGenerator(
                    model=args.grok_model,
                    timeout_seconds=args.api_timeout,
                )
        except (GeminiConfigurationError, GrokConfigurationError) as exc:
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
    except (GeminiGenerationError, GrokGenerationError) as exc:
        raise SystemExit(f"Generation failed: {exc}") from exc

    print_final_response(answer)
    print_debug_information(answer, results)

    if args.show_evidence:
        print(f"\n{DIVIDER}\n")
        print_results(results)


if __name__ == "__main__":
    main()
