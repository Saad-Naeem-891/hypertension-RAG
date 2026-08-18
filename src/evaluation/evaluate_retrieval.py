"""Evaluate hybrid retrieval against chunk-ID ground truth and log every run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import shlex
import subprocess
import time
from typing import Any, Protocol, Sequence

from src.embedding.embed_chunks import DEFAULT_CHUNKS_DIRECTORY, load_chunks
from src.retrieval.hybrid_retriever import (
    BM25_B,
    BM25_K1,
    DEFAULT_CANDIDATE_K,
    DEFAULT_RRF_K,
    BM25Retriever,
    HybridRetriever,
)
from src.retrieval.semantic_retriever import QUERY_PREFIX, SemanticRetriever
from src.reranking import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RerankedHybridRetriever,
)
from src.vector_store.qdrant_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATABASE_PATH,
    DEFAULT_MANIFEST_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "artifacts" / "truth_table" / "potassium intake trustable.json"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "evaluation"
RUNS_FILENAME = "evaluation_runs.csv"
QUESTION_RESULTS_FILENAME = "evaluation_question_results.csv"
SCHEMA_VERSION = 2
DEFAULT_TOP_K_VALUES = (5, 10, 20)
FLATTENED_K_VALUES = (3, 5, 10)


RUN_FIELDS = [
    "schema_version",
    "run_id",
    "rerun_of",
    "started_at_utc",
    "finished_at_utc",
    "duration_seconds",
    "status",
    "error",
    "run_command",
    "git_commit",
    "git_dirty",
    "source_fingerprint",
    "python_version",
    "sentence_transformers_version",
    "qdrant_client_version",
    "ground_truth_path",
    "ground_truth_sha256",
    "question_count",
    "relevance_label_count",
    "chunks_directory",
    "chunk_count",
    "chunks_fingerprint",
    "manifest_path",
    "manifest_sha256",
    "embedding_model",
    "embedding_dimension",
    "query_prefix",
    "normalized_embeddings",
    "device",
    "qdrant_path",
    "qdrant_collection",
    "retriever_type",
    "reranker_enabled",
    "reranker_model",
    "reranker_batch_size",
    "candidate_k",
    "rrf_k",
    "bm25_k1",
    "bm25_b",
    "top_k_values",
    "precision_at_3",
    "recall_at_3",
    "hit_rate_at_3",
    "mrr_at_3",
    "ndcg_at_3",
    "precision_at_5",
    "recall_at_5",
    "hit_rate_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "precision_at_10",
    "recall_at_10",
    "hit_rate_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "metrics_json",
    "details_file",
]


QUESTION_RESULT_FIELDS = [
    "schema_version",
    "run_id",
    "question_index",
    "question",
    "k",
    "relevant_chunk_ids",
    "retrieved_chunk_ids",
    "hit_chunk_ids",
    "missed_chunk_ids",
    "dense_ranks",
    "bm25_ranks",
    "hybrid_scores",
    "pre_rerank_ranks",
    "rerank_scores",
    "precision",
    "recall",
    "hit",
    "reciprocal_rank",
    "ndcg",
]


class EvaluationRetriever(Protocol):
    """Retrieval interface required by the evaluator."""

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        candidate_k: int,
    ) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """One evaluation question and its binary relevance judgments."""

    question: str
    relevant_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Binary retrieval metrics for one question or an aggregate."""

    precision: float
    recall: float
    hit: float
    reciprocal_rank: float
    ndcg: float


@dataclass(frozen=True, slots=True)
class QuestionEvaluation:
    """One question's results at a particular retrieval cutoff."""

    question_index: int
    question: str
    k: int
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    hit_chunk_ids: tuple[str, ...]
    missed_chunk_ids: tuple[str, ...]
    dense_ranks: tuple[int | None, ...]
    bm25_ranks: tuple[int | None, ...]
    hybrid_scores: tuple[float | None, ...]
    pre_rerank_ranks: tuple[int | None, ...]
    rerank_scores: tuple[float | None, ...]
    metrics: EvaluationMetrics


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Complete per-question results and mean metrics for an evaluation run."""

    question_results: tuple[QuestionEvaluation, ...]
    mean_metrics: dict[int, EvaluationMetrics]


def _normalized_top_k_values(values: Sequence[int]) -> tuple[int, ...]:
    top_k_values = tuple(sorted(set(values)))
    if not top_k_values or any(value < 1 for value in top_k_values):
        raise ValueError("Every top-k value must be at least 1")
    return top_k_values


def load_ground_truth(
    ground_truth_path: str | Path,
    *,
    valid_chunk_ids: set[str] | None = None,
) -> list[EvaluationExample]:
    """Load and validate the question-to-relevant-chunk-ID judgments."""

    path = Path(ground_truth_path).expanduser().resolve()
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Ground truth must be a non-empty JSON array")

    examples: list[EvaluationExample] = []
    seen_questions: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Ground-truth item {index} must be an object")
        question = record.get("question")
        relevant_ids = record.get("relevant_chunk_ids")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Ground-truth item {index} has an invalid question")
        cleaned_question = question.strip()
        if cleaned_question in seen_questions:
            raise ValueError(f"Duplicate evaluation question: {cleaned_question}")
        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in relevant_ids)
        ):
            raise ValueError(
                f"Ground-truth item {index} must have non-empty relevant_chunk_ids"
            )
        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError(f"Ground-truth item {index} contains duplicate chunk IDs")
        if valid_chunk_ids is not None:
            missing_ids = sorted(set(relevant_ids) - valid_chunk_ids)
            if missing_ids:
                raise ValueError(
                    f"Ground-truth item {index} contains unknown chunk IDs: {missing_ids}"
                )

        seen_questions.add(cleaned_question)
        examples.append(
            EvaluationExample(cleaned_question, tuple(relevant_ids))
        )
    return examples


def calculate_metrics(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    *,
    k: int,
) -> EvaluationMetrics:
    """Calculate binary Precision, Recall, Hit, MRR and nDCG at K."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant_chunk_ids:
        raise ValueError("At least one relevant chunk ID is required")

    retrieved = list(retrieved_chunk_ids[:k])
    if len(retrieved) != len(set(retrieved)):
        raise ValueError("Retrieved chunk IDs must be unique")
    relevant = set(relevant_chunk_ids)
    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(retrieved, start=1)
        if chunk_id in relevant
    ]
    hit_count = len(relevant_ranks)
    precision = hit_count / k
    recall = hit_count / len(relevant)
    hit = float(bool(relevant_ranks))
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return EvaluationMetrics(precision, recall, hit, reciprocal_rank, ndcg)


def evaluate_retriever(
    examples: Sequence[EvaluationExample],
    retriever: EvaluationRetriever,
    *,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> EvaluationReport:
    """Retrieve once per question and evaluate every configured cutoff."""

    if not examples:
        raise ValueError("At least one evaluation example is required")
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")
    cutoffs = _normalized_top_k_values(top_k_values)
    maximum_k = max(cutoffs)
    question_results: list[QuestionEvaluation] = []

    for question_index, example in enumerate(examples, start=1):
        retrieved = list(
            retriever.retrieve(
                example.question,
                top_k=maximum_k,
                candidate_k=max(candidate_k, maximum_k),
            )
        )
        retrieved_ids = [str(result.chunk_id) for result in retrieved]
        if len(retrieved_ids) != len(set(retrieved_ids)):
            raise ValueError(
                f"Retriever returned duplicate chunks for question {question_index}"
            )

        for k in cutoffs:
            top_results = retrieved[:k]
            top_ids = tuple(str(result.chunk_id) for result in top_results)
            relevant = set(example.relevant_chunk_ids)
            hit_ids = tuple(chunk_id for chunk_id in top_ids if chunk_id in relevant)
            missed_ids = tuple(
                chunk_id
                for chunk_id in example.relevant_chunk_ids
                if chunk_id not in set(top_ids)
            )
            question_results.append(
                QuestionEvaluation(
                    question_index=question_index,
                    question=example.question,
                    k=k,
                    relevant_chunk_ids=example.relevant_chunk_ids,
                    retrieved_chunk_ids=top_ids,
                    hit_chunk_ids=hit_ids,
                    missed_chunk_ids=missed_ids,
                    dense_ranks=tuple(
                        getattr(result, "dense_rank", None) for result in top_results
                    ),
                    bm25_ranks=tuple(
                        getattr(result, "bm25_rank", None) for result in top_results
                    ),
                    hybrid_scores=tuple(
                        (
                            float(result.hybrid_score)
                            if getattr(result, "hybrid_score", None) is not None
                            else None
                        )
                        for result in top_results
                    ),
                    pre_rerank_ranks=tuple(
                        getattr(result, "original_rank", None)
                        for result in top_results
                    ),
                    rerank_scores=tuple(
                        (
                            float(result.rerank_score)
                            if getattr(result, "rerank_score", None) is not None
                            else None
                        )
                        for result in top_results
                    ),
                    metrics=calculate_metrics(
                        top_ids,
                        example.relevant_chunk_ids,
                        k=k,
                    ),
                )
            )

    mean_metrics: dict[int, EvaluationMetrics] = {}
    for k in cutoffs:
        metrics = [
            result.metrics for result in question_results if result.k == k
        ]
        mean_metrics[k] = EvaluationMetrics(
            precision=sum(metric.precision for metric in metrics) / len(metrics),
            recall=sum(metric.recall for metric in metrics) / len(metrics),
            hit=sum(metric.hit for metric in metrics) / len(metrics),
            reciprocal_rank=(
                sum(metric.reciprocal_rank for metric in metrics) / len(metrics)
            ),
            ndcg=sum(metric.ndcg for metric in metrics) / len(metrics),
        )
    return EvaluationReport(tuple(question_results), mean_metrics)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_files(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _chunks_fingerprint(chunks_directory: Path) -> str:
    json_files = list(chunks_directory.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No chunk JSON files found in: {chunks_directory}")
    return _fingerprint_files(json_files, chunks_directory)


def _source_fingerprint() -> str:
    paths = list((PROJECT_ROOT / "src").rglob("*.py"))
    paths.extend(
        path
        for path in (PROJECT_ROOT / "main.py", PROJECT_ROOT / "requirements.txt")
        if path.is_file()
    )
    return _fingerprint_files(paths, PROJECT_ROOT)


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _new_run_id(now: datetime) -> str:
    return now.strftime("eval_%Y%m%dT%H%M%S%fZ")


def _append_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Append rows and migrate existing files when new optional fields are added."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            existing_header = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_header != list(fieldnames):
            unexpected_fields = sorted(set(existing_header) - set(fieldnames))
            if unexpected_fields:
                raise ValueError(
                    f"Existing CSV has unsupported fields {unexpected_fields}: {path}"
                )
            temporary_path = path.with_name(f".{path.name}.schema-migration")
            with temporary_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=fieldnames,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(existing_rows)
            temporary_path.replace(path)

    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _question_rows(run_id: str, report: EvaluationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in report.question_results:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "question_index": result.question_index,
                "question": result.question,
                "k": result.k,
                "relevant_chunk_ids": _json_cell(result.relevant_chunk_ids),
                "retrieved_chunk_ids": _json_cell(result.retrieved_chunk_ids),
                "hit_chunk_ids": _json_cell(result.hit_chunk_ids),
                "missed_chunk_ids": _json_cell(result.missed_chunk_ids),
                "dense_ranks": _json_cell(result.dense_ranks),
                "bm25_ranks": _json_cell(result.bm25_ranks),
                "hybrid_scores": _json_cell(result.hybrid_scores),
                "pre_rerank_ranks": _json_cell(result.pre_rerank_ranks),
                "rerank_scores": _json_cell(result.rerank_scores),
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "hit": result.metrics.hit,
                "reciprocal_rank": result.metrics.reciprocal_rank,
                "ndcg": result.metrics.ndcg,
            }
        )
    return rows


def _add_report_metrics(run_row: dict[str, Any], report: EvaluationReport) -> None:
    metrics_json = {
        str(k): asdict(metrics) for k, metrics in report.mean_metrics.items()
    }
    run_row["metrics_json"] = _json_cell(metrics_json)
    for k in FLATTENED_K_VALUES:
        metrics = report.mean_metrics.get(k)
        if metrics is None:
            continue
        run_row[f"precision_at_{k}"] = metrics.precision
        run_row[f"recall_at_{k}"] = metrics.recall
        run_row[f"hit_rate_at_{k}"] = metrics.hit
        run_row[f"mrr_at_{k}"] = metrics.reciprocal_rank
        run_row[f"ndcg_at_{k}"] = metrics.ndcg


def _empty_run_row(run_id: str, started_at: datetime) -> dict[str, Any]:
    row = {field: "" for field in RUN_FIELDS}
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "started_at_utc": started_at.isoformat(),
            "status": "running",
        }
    )
    return row


def _canonical_command(config: dict[str, Any]) -> str:
    command = [
        "conda",
        "run",
        "-n",
        "student_rag",
        "python",
        "-m",
        "src.evaluation.evaluate_retrieval",
        "--ground-truth",
        str(config["ground_truth_path"]),
        "--chunks-directory",
        str(config["chunks_directory"]),
        "--manifest-path",
        str(config["manifest_path"]),
        "--database-path",
        str(config["database_path"]),
        "--collection-name",
        str(config["collection_name"]),
        "--top-k",
        *(str(value) for value in config["top_k_values"]),
        "--candidate-k",
        str(config["candidate_k"]),
        "--rrf-k",
        str(config["rrf_k"]),
        "--bm25-k1",
        str(config["bm25_k1"]),
        "--bm25-b",
        str(config["bm25_b"]),
        "--reranker-model",
        str(config["reranker_model"]),
        "--reranker-batch-size",
        str(config["reranker_batch_size"]),
        "--device",
        str(config["device"]),
        "--output-directory",
        str(config["output_directory"]),
    ]
    if not config["reranker_enabled"]:
        command.append("--no-reranker")
    return shlex.join(command)


def _load_run_row(run_id: str, runs_path: Path) -> dict[str, str]:
    if not runs_path.is_file():
        raise FileNotFoundError(f"Evaluation run history does not exist: {runs_path}")
    with runs_path.open("r", encoding="utf-8", newline="") as source:
        matches = [row for row in csv.DictReader(source) if row.get("run_id") == run_id]
    if not matches:
        raise ValueError(f"Evaluation run ID was not found: {run_id}")
    return matches[-1]


def _configuration_from_previous_run(
    previous: dict[str, str],
    output_directory: Path,
) -> dict[str, Any]:
    return {
        "ground_truth_path": Path(previous["ground_truth_path"]),
        "chunks_directory": Path(previous["chunks_directory"]),
        "manifest_path": Path(previous["manifest_path"]),
        "database_path": Path(previous["qdrant_path"]),
        "collection_name": previous["qdrant_collection"],
        "top_k_values": _normalized_top_k_values(
            tuple(json.loads(previous["top_k_values"]))
        ),
        "candidate_k": int(previous["candidate_k"]),
        "rrf_k": int(previous["rrf_k"]),
        "bm25_k1": float(previous["bm25_k1"]),
        "bm25_b": float(previous["bm25_b"]),
        "reranker_enabled": str(previous.get("reranker_enabled", "")).lower()
        in {"1", "true", "yes"},
        "reranker_model": previous.get("reranker_model") or DEFAULT_RERANKER_MODEL,
        "reranker_batch_size": int(
            previous.get("reranker_batch_size") or DEFAULT_RERANKER_BATCH_SIZE
        ),
        "device": previous["device"],
        "output_directory": output_directory,
    }


def _reproducibility_warnings(
    previous: dict[str, str],
    current: dict[str, Any],
) -> list[str]:
    checks = {
        "ground truth": "ground_truth_sha256",
        "chunks": "chunks_fingerprint",
        "embedding manifest": "manifest_sha256",
        "source code": "source_fingerprint",
        "Git commit": "git_commit",
    }
    return [
        f"{label} changed since {previous['run_id']}"
        for label, field in checks.items()
        if previous.get(field) and str(current.get(field)) != previous[field]
    ]


def _print_report(run_id: str, report: EvaluationReport) -> None:
    print(f"\nEvaluation run: {run_id}")
    print("\nMean retrieval metrics")
    print("K   Precision   Recall   Hit Rate   MRR      nDCG")
    for k, metrics in report.mean_metrics.items():
        print(
            f"{k:<3} {metrics.precision:>9.4f} "
            f"{metrics.recall:>8.4f} {metrics.hit:>10.4f} "
            f"{metrics.reciprocal_rank:>8.4f} {metrics.ndcg:>8.4f}"
        )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--chunks-directory", type=Path, default=DEFAULT_CHUNKS_DIRECTORY)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--database-path", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOP_K_VALUES),
        help=(
            "Evaluation cutoffs "
            f"(default: {' '.join(str(k) for k in DEFAULT_TOP_K_VALUES)})"
        ),
    )
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--bm25-k1", type=float, default=BM25_K1)
    parser.add_argument("--bm25-b", type=float, default=BM25_B)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=DEFAULT_RERANKER_BATCH_SIZE,
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Evaluate the hybrid RRF baseline without cross-encoder reranking",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--rerun",
        metavar="RUN_ID",
        help="Reuse the recorded configuration from an earlier evaluation run",
    )
    return parser


def _configuration_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_directory = args.output_directory.expanduser().resolve()
    if args.rerun:
        previous = _load_run_row(args.rerun, output_directory / RUNS_FILENAME)
        return _configuration_from_previous_run(previous, output_directory)
    return {
        "ground_truth_path": args.ground_truth.expanduser().resolve(),
        "chunks_directory": args.chunks_directory.expanduser().resolve(),
        "manifest_path": args.manifest_path.expanduser().resolve(),
        "database_path": args.database_path.expanduser().resolve(),
        "collection_name": args.collection_name,
        "top_k_values": _normalized_top_k_values(args.top_k),
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "reranker_enabled": not args.no_reranker,
        "reranker_model": args.reranker_model,
        "reranker_batch_size": args.reranker_batch_size,
        "device": args.device,
        "output_directory": output_directory,
    }


def main() -> None:
    args = _argument_parser().parse_args()
    config = _configuration_from_args(args)
    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    run_id = _new_run_id(started_at)
    run_row = _empty_run_row(run_id, started_at)
    output_directory = config["output_directory"]
    runs_path = output_directory / RUNS_FILENAME
    details_path = output_directory / QUESTION_RESULTS_FILENAME
    dense_retriever: SemanticRetriever | None = None
    run_row.update(
        {
            "rerun_of": args.rerun or "",
            "run_command": _canonical_command(config),
            "ground_truth_path": str(config["ground_truth_path"]),
            "chunks_directory": str(config["chunks_directory"]),
            "manifest_path": str(config["manifest_path"]),
            "device": config["device"],
            "qdrant_path": str(config["database_path"]),
            "qdrant_collection": config["collection_name"],
            "retriever_type": "hybrid_dense_bm25_rrf",
            "reranker_enabled": config["reranker_enabled"],
            "reranker_model": (
                config["reranker_model"] if config["reranker_enabled"] else ""
            ),
            "reranker_batch_size": (
                config["reranker_batch_size"] if config["reranker_enabled"] else ""
            ),
            "candidate_k": config["candidate_k"],
            "rrf_k": config["rrf_k"],
            "bm25_k1": config["bm25_k1"],
            "bm25_b": config["bm25_b"],
            "top_k_values": _json_cell(config["top_k_values"]),
            "details_file": str(details_path),
        }
    )

    try:
        chunks = load_chunks(config["chunks_directory"])
        valid_chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        examples = load_ground_truth(
            config["ground_truth_path"],
            valid_chunk_ids=valid_chunk_ids,
        )
        manifest = json.loads(config["manifest_path"].read_text(encoding="utf-8"))
        git_status = _git_value("status", "--porcelain")
        metadata = {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_status),
            "source_fingerprint": _source_fingerprint(),
            "ground_truth_sha256": _sha256_file(config["ground_truth_path"]),
            "chunks_fingerprint": _chunks_fingerprint(config["chunks_directory"]),
            "manifest_sha256": _sha256_file(config["manifest_path"]),
        }
        run_row.update(
            {
                **metadata,
                "python_version": platform.python_version(),
                "sentence_transformers_version": _package_version(
                    "sentence-transformers"
                ),
                "qdrant_client_version": _package_version("qdrant-client"),
                "question_count": len(examples),
                "relevance_label_count": sum(
                    len(example.relevant_chunk_ids) for example in examples
                ),
                "chunk_count": len(chunks),
                "embedding_model": manifest.get("model_name"),
                "embedding_dimension": manifest.get("embedding_dimension"),
                "query_prefix": manifest.get("query_prefix", QUERY_PREFIX),
                "normalized_embeddings": manifest.get("normalized"),
            }
        )

        if args.rerun:
            previous = _load_run_row(args.rerun, runs_path)
            for warning in _reproducibility_warnings(previous, metadata):
                print(f"WARNING: {warning}")
        if run_row["git_dirty"]:
            print("WARNING: Git worktree is dirty; exact source reproduction requires a clean commit.")

        dense_retriever = SemanticRetriever(
            database_path=config["database_path"],
            collection_name=config["collection_name"],
            manifest_path=config["manifest_path"],
            device=config["device"],
        )
        bm25_retriever = BM25Retriever(
            chunks,
            k1=config["bm25_k1"],
            b=config["bm25_b"],
        )
        hybrid_retriever = HybridRetriever(
            chunks=chunks,
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            rrf_k=config["rrf_k"],
        )
        retriever: EvaluationRetriever = hybrid_retriever
        if config["reranker_enabled"]:
            reranker = CrossEncoderReranker(
                config["reranker_model"],
                device=config["device"],
                batch_size=config["reranker_batch_size"],
            )
            retriever = RerankedHybridRetriever(
                hybrid_retriever=hybrid_retriever,
                reranker=reranker,
            )
            run_row["retriever_type"] = "hybrid_dense_bm25_rrf_cross_encoder"
        report = evaluate_retriever(
            examples,
            retriever,
            top_k_values=config["top_k_values"],
            candidate_k=config["candidate_k"],
        )
        run_row["status"] = "success"
        _add_report_metrics(run_row, report)
        _append_csv(details_path, QUESTION_RESULT_FIELDS, _question_rows(run_id, report))
        _print_report(run_id, report)
    except BaseException as error:
        run_row["status"] = "failed"
        run_row["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if dense_retriever is not None:
            dense_retriever.close()
        finished_at = datetime.now(timezone.utc)
        run_row["finished_at_utc"] = finished_at.isoformat()
        run_row["duration_seconds"] = round(time.perf_counter() - start_time, 6)
        _append_csv(runs_path, RUN_FIELDS, [run_row])
        print(f"Run history: {runs_path}")
        if run_row["status"] == "success":
            print(f"Question details: {details_path}")


if __name__ == "__main__":
    main()
