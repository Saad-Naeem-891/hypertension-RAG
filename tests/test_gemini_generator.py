from types import SimpleNamespace
import json
import os
import unittest
from unittest.mock import patch

from src.generation.gemini_generator import (
    GeminiConfigurationError,
    GeminiGenerationError,
    GeminiGenerator,
)
from src.reranking import RerankedChunk


def evidence_chunk() -> RerankedChunk:
    return RerankedChunk(
        rank=1,
        rerank_score=0.9,
        original_rank=1,
        hybrid_score=0.03,
        dense_rank=1,
        bm25_rank=1,
        retrieved_by="Both",
        chunk_id="potassium_chunk_001",
        text="The recommendation applies to adults.",
        document_name="Potassium intake",
        section_title="Recommendations",
        page_start=10,
        page_end=10,
    )


class FakeInteractions:
    def __init__(self, output_text: str | None = None) -> None:
        if output_text is None:
            output_text = json.dumps(
                {
                    "recommendation": "The recommendation applies to adults.",
                    "supporting_evidence": [
                        {
                            "statement": "The source explicitly applies to adults.",
                            "chunk_ids": ["potassium_chunk_001"],
                        }
                    ],
                    "confidence": "high",
                    "safety_message": "Guideline evidence only.",
                }
            )
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, interactions: FakeInteractions | None = None) -> None:
        self.interactions = interactions or FakeInteractions()


class GeminiGeneratorTests(unittest.TestCase):
    def test_generate_uses_stateless_interactions_api(self) -> None:
        client = FakeClient()
        generator = GeminiGenerator(model="test-gemini", client=client)

        answer = generator.generate("Who does it apply to?", [evidence_chunk()])

        self.assertEqual(answer.recommendation, "The recommendation applies to adults.")
        self.assertEqual(answer.confidence, "High")
        self.assertEqual(answer.citations[0].chunk_id, "potassium_chunk_001")
        self.assertEqual(answer.provider, "Google Gemini")
        self.assertEqual(answer.model, "test-gemini")
        request = client.interactions.calls[0]
        self.assertEqual(request["model"], "test-gemini")
        self.assertIs(request["store"], False)
        self.assertIn("evidence-grounded assistant", request["system_instruction"])
        self.assertIn("[potassium_chunk_001]", request["input"])
        self.assertEqual(
            request["response_format"]["mime_type"],
            "application/json",
        )

    def test_missing_api_key_has_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GeminiConfigurationError, "GEMINI_API_KEY"):
                GeminiGenerator()

    def test_empty_api_response_is_rejected(self) -> None:
        generator = GeminiGenerator(
            client=FakeClient(FakeInteractions(output_text="")),
        )

        with self.assertRaisesRegex(GeminiGenerationError, "empty answer"):
            generator.generate("Question", [evidence_chunk()])


if __name__ == "__main__":
    unittest.main()
