import json
from types import SimpleNamespace
import unittest

from pydantic import ValidationError

from src.api.app import ChatRequest, _metric_rows, _safe_evaluation_row, chat


class ApiContractTests(unittest.TestCase):
    def test_chat_request_strips_message(self) -> None:
        request = ChatRequest(message="  What is the sodium recommendation?  ")

        self.assertEqual(request.message, "What is the sodium recommendation?")

    def test_chat_request_rejects_whitespace(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest(message="   ")

    def test_metric_rows_parse_dynamic_cutoffs(self) -> None:
        metrics = _metric_rows(
            json.dumps(
                {
                    "5": {
                        "precision": 0.2,
                        "recall": 0.7,
                        "hit": 0.9,
                        "reciprocal_rank": 0.8,
                        "ndcg": 0.75,
                    }
                }
            )
        )

        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].cutoff, 5)
        self.assertEqual(metrics[0].mrr, 0.8)

    def test_evaluation_response_does_not_expose_internal_path(self) -> None:
        response = _safe_evaluation_row(
            {
                "run_id": "eval_1",
                "ground_truth_path": "/private/project/truth.json",
                "metrics_json": "{}",
            }
        )

        self.assertEqual(response.ground_truth_name, "truth.json")
        self.assertNotIn("/private/project", response.model_dump_json())

    def test_safety_guard_stops_before_retrieval(self) -> None:
        class RetrieverThatMustNotRun:
            def retrieve(self, *_args, **_kwargs):
                raise AssertionError("retrieval should not run")

        response = chat(
            ChatRequest(message="I have chest pain and cannot breathe"),
            RetrieverThatMustNotRun(),
        )

        self.assertEqual(response.confidence, "Insufficient Evidence")
        self.assertEqual(response.citations, [])
        self.assertIn("emergency", response.safety_message.lower())

    def test_low_evidence_confidence_stops_before_generation(self) -> None:
        class LowConfidenceRetriever:
            def retrieve(self, *_args, **_kwargs):
                return [SimpleNamespace(rerank_score=0.0)]

        response = chat(
            ChatRequest(message="What is the sodium recommendation?"),
            LowConfidenceRetriever(),
        )

        self.assertEqual(response.confidence, "Insufficient Evidence")
        self.assertEqual(response.evidence_confidence_percentage, 50.0)
        self.assertEqual(response.citations, [])


if __name__ == "__main__":
    unittest.main()
