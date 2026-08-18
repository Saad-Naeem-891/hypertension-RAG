import csv
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import json
import unittest

from src.evaluation.evaluate_retrieval import (
    QUESTION_RESULT_FIELDS,
    RUN_FIELDS,
    EvaluationExample,
    _add_report_metrics,
    _append_csv,
    _configuration_from_previous_run,
    _empty_run_row,
    _question_rows,
    calculate_metrics,
    evaluate_retriever,
    load_ground_truth,
)


class FakeRetriever:
    def __init__(self, results_by_question: dict[str, list[str]]) -> None:
        self.results_by_question = results_by_question
        self.calls: list[tuple[str, int, int]] = []

    def retrieve(self, question: str, *, top_k: int, candidate_k: int) -> list:
        self.calls.append((question, top_k, candidate_k))
        return [
            SimpleNamespace(
                chunk_id=chunk_id,
                dense_rank=rank,
                bm25_rank=rank + 1,
                hybrid_score=1 / (60 + rank),
            )
            for rank, chunk_id in enumerate(
                self.results_by_question[question][:top_k],
                start=1,
            )
        ]


class EvaluationMetricsTests(unittest.TestCase):
    def test_metrics_use_binary_relevance_and_rank(self) -> None:
        metrics = calculate_metrics(["x", "a", "b"], ["a", "b"], k=3)

        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.hit, 1.0)
        self.assertEqual(metrics.reciprocal_rank, 0.5)
        expected_dcg = 1 / 1.584962500721156 + 1 / 2
        expected_ideal_dcg = 1 + 1 / 1.584962500721156
        self.assertAlmostEqual(metrics.ndcg, expected_dcg / expected_ideal_dcg)

    def test_evaluator_retrieves_once_per_question_for_largest_k(self) -> None:
        examples = [
            EvaluationExample("q1", ("a", "b")),
            EvaluationExample("q2", ("e",)),
        ]
        retriever = FakeRetriever(
            {
                "q1": ["x", "a", "b", "c", "d"],
                "q2": ["e", "x", "y", "z", "a"],
            }
        )

        report = evaluate_retriever(
            examples,
            retriever,
            top_k_values=[3, 5],
            candidate_k=7,
        )

        self.assertEqual(retriever.calls, [("q1", 5, 7), ("q2", 5, 7)])
        self.assertEqual(len(report.question_results), 4)
        self.assertAlmostEqual(report.mean_metrics[3].precision, 0.5)
        self.assertEqual(report.mean_metrics[3].hit, 1.0)
        first = report.question_results[0]
        self.assertEqual(first.hit_chunk_ids, ("a", "b"))
        self.assertEqual(first.dense_ranks, (1, 2, 3))
        self.assertEqual(first.bm25_ranks, (2, 3, 4))


class GroundTruthTests(unittest.TestCase):
    def test_unknown_ground_truth_chunk_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "truth.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "question": "Question?",
                            "relevant_chunk_ids": ["missing"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown chunk IDs"):
                load_ground_truth(path, valid_chunk_ids={"existing"})


class CsvHistoryTests(unittest.TestCase):
    def test_run_and_question_history_are_append_only(self) -> None:
        examples = [EvaluationExample("q1", ("a",))]
        retriever = FakeRetriever({"q1": ["a", "b", "c"]})
        report = evaluate_retriever(
            examples,
            retriever,
            top_k_values=[3],
            candidate_k=3,
        )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs_path = root / "runs.csv"
            details_path = root / "details.csv"
            started_at = datetime.now(timezone.utc)

            for run_id in ("run_1", "run_2"):
                run_row = _empty_run_row(run_id, started_at)
                run_row["status"] = "success"
                _add_report_metrics(run_row, report)
                _append_csv(runs_path, RUN_FIELDS, [run_row])
                _append_csv(
                    details_path,
                    QUESTION_RESULT_FIELDS,
                    _question_rows(run_id, report),
                )

            with runs_path.open("r", encoding="utf-8", newline="") as source:
                run_rows = list(csv.DictReader(source))
            with details_path.open("r", encoding="utf-8", newline="") as source:
                detail_rows = list(csv.DictReader(source))

            self.assertEqual([row["run_id"] for row in run_rows], ["run_1", "run_2"])
            self.assertEqual(len(detail_rows), 2)
            self.assertEqual(detail_rows[0]["retrieved_chunk_ids"], '["a","b","c"]')
            self.assertEqual(detail_rows[0]["rerank_scores"], "[null,null,null]")
            self.assertEqual(float(run_rows[0]["precision_at_3"]), 1 / 3)

    def test_csv_schema_additions_migrate_existing_rows(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "history.csv"
            path.write_text("run_id,status\nold,success\n", encoding="utf-8")

            _append_csv(
                path,
                ["run_id", "status", "reranker_model"],
                [{"run_id": "new", "status": "success", "reranker_model": "model"}],
            )

            with path.open("r", encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))

            self.assertEqual(rows[0]["run_id"], "old")
            self.assertEqual(rows[0]["reranker_model"], "")
            self.assertEqual(rows[1]["reranker_model"], "model")

    def test_previous_run_configuration_can_be_reloaded(self) -> None:
        previous = {
            "ground_truth_path": "/project/truth.json",
            "chunks_directory": "/project/chunks",
            "manifest_path": "/project/manifest.json",
            "qdrant_path": "/project/qdrant",
            "qdrant_collection": "guidelines",
            "top_k_values": "[3,5,10]",
            "candidate_k": "20",
            "rrf_k": "60",
            "bm25_k1": "1.5",
            "bm25_b": "0.75",
            "device": "cpu",
        }

        config = _configuration_from_previous_run(previous, Path("/new/output"))

        self.assertEqual(config["top_k_values"], (3, 5, 10))
        self.assertEqual(config["candidate_k"], 20)
        self.assertEqual(config["rrf_k"], 60)
        self.assertEqual(config["bm25_k1"], 1.5)
        self.assertFalse(config["reranker_enabled"])
        self.assertEqual(config["collection_name"], "guidelines")
        self.assertEqual(config["output_directory"], Path("/new/output"))

    def test_reranked_run_configuration_can_be_reloaded(self) -> None:
        previous = {
            "ground_truth_path": "/project/truth.json",
            "chunks_directory": "/project/chunks",
            "manifest_path": "/project/manifest.json",
            "qdrant_path": "/project/qdrant",
            "qdrant_collection": "guidelines",
            "top_k_values": "[5,10,20]",
            "candidate_k": "25",
            "rrf_k": "60",
            "bm25_k1": "1.5",
            "bm25_b": "0.75",
            "reranker_enabled": "True",
            "reranker_model": "cross-encoder/test-model",
            "reranker_batch_size": "8",
            "device": "cpu",
        }

        config = _configuration_from_previous_run(previous, Path("/new/output"))

        self.assertTrue(config["reranker_enabled"])
        self.assertEqual(config["reranker_model"], "cross-encoder/test-model")
        self.assertEqual(config["reranker_batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
