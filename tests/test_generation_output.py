from contextlib import redirect_stdout
from io import StringIO
import json
import unittest

from main import print_debug_information, print_final_response
from src.generation.common import GeneratedAnswer, parse_generated_answer
from src.reranking import RerankedChunk


def chunk(chunk_id: str, page: int) -> RerankedChunk:
    return RerankedChunk(
        rank=page,
        rerank_score=1.0,
        original_rank=page,
        hybrid_score=0.03,
        dense_rank=page,
        bm25_rank=page,
        retrieved_by="Both",
        chunk_id=chunk_id,
        text=f"Evidence from {chunk_id}",
        document_name="WHO Guideline",
        section_title="Recommendations",
        page_start=page,
        page_end=page,
    )


def structured_json(*, chunk_id: str = "used") -> str:
    return json.dumps(
        {
            "recommendation": "Follow the guideline recommendation.",
            "supporting_evidence": [
                {
                    "statement": "The guideline directly supports this recommendation.",
                    "chunk_ids": [chunk_id],
                }
            ],
            "confidence": "high",
            "safety_message": "This is guideline evidence, not individualized advice.",
        }
    )


class StructuredAnswerTests(unittest.TestCase):
    def test_only_chunks_used_by_supporting_evidence_become_citations(self) -> None:
        supplied = [chunk("used", 4), chunk("unused", 9)]

        answer = parse_generated_answer(
            structured_json(),
            supplied,
            provider="Test Provider",
            model="test-model",
        )

        self.assertEqual([item.chunk_id for item in answer.citations], ["used"])
        self.assertEqual(answer.citations[0].document_name, "WHO Guideline")
        self.assertEqual(answer.citations[0].page_start, 4)
        self.assertEqual(answer.evidence_chunk_ids, ("used", "unused"))

    def test_chunk_not_supplied_to_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "was not supplied"):
            parse_generated_answer(
                structured_json(chunk_id="invented"),
                [chunk("used", 4)],
                provider="Test Provider",
                model="test-model",
            )

    def test_insufficient_answer_has_no_support_or_citations(self) -> None:
        payload = json.dumps(
            {
                "recommendation": "Maybe.",
                "supporting_evidence": [],
                "confidence": "insufficient_evidence",
                "safety_message": "Guideline evidence only.",
            }
        )

        answer = parse_generated_answer(
            payload,
            [chunk("available", 1)],
            provider="Test Provider",
            model="test-model",
        )

        self.assertEqual(answer.confidence, "Insufficient Evidence")
        self.assertIn("insufficient", answer.recommendation.lower())
        self.assertEqual(answer.supporting_evidence, ())
        self.assertEqual(answer.citations, ())

    def test_terminal_sections_and_debug_information_are_separate(self) -> None:
        supplied = [chunk("used", 4), chunk("unused", 9)]
        answer = parse_generated_answer(
            structured_json(),
            supplied,
            provider="Test Provider",
            model="test-model",
        )
        output = StringIO()

        with redirect_stdout(output):
            print_final_response(answer)
            print_debug_information(answer, supplied)

        rendered = output.getvalue()
        self.assertIn("FINAL RESPONSE", rendered)
        self.assertIn("Recommendation:", rendered)
        self.assertIn("Supporting Evidence:", rendered)
        self.assertIn("Citations:", rendered)
        self.assertIn("Confidence:", rendered)
        self.assertIn("Safety:", rendered)
        self.assertIn("DEBUG / RETRIEVAL INFORMATION", rendered)
        self.assertIn("Document: WHO Guideline", rendered)
        self.assertIn("1. used", rendered)
        self.assertIn("2. unused", rendered)


if __name__ == "__main__":
    unittest.main()
