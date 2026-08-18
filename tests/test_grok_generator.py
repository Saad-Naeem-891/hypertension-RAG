from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

from src.generation.grok_generator import (
    GrokConfigurationError,
    GrokGenerationError,
    GrokGenerator,
    build_grounded_prompt,
)
from src.reranking import RerankedChunk


def evidence_chunk(
    chunk_id: str = "sodium_chunk_001",
    text: str = "WHO recommends reducing sodium intake.",
) -> RerankedChunk:
    return RerankedChunk(
        rank=1,
        rerank_score=0.9,
        original_rank=2,
        hybrid_score=0.03,
        dense_rank=2,
        bm25_rank=1,
        retrieved_by="Both",
        chunk_id=chunk_id,
        text=text,
        document_name="Sodium intake for adults and children",
        section_title="Recommendations",
        page_start=17,
        page_end=18,
    )


class FakeResponses:
    def __init__(self, *, output_text: str = "Grounded answer [sodium_chunk_001]") -> None:
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, responses: FakeResponses | None = None) -> None:
        self.responses = responses or FakeResponses()


class GroundedPromptTests(unittest.TestCase):
    def test_prompt_contains_question_citation_metadata_and_original_text(self) -> None:
        prompt = build_grounded_prompt(
            "What is recommended?",
            [evidence_chunk()],
        )

        self.assertIn("Question:\nWhat is recommended?", prompt)
        self.assertIn("[sodium_chunk_001]", prompt)
        self.assertIn("Pages: 17-18", prompt)
        self.assertIn("WHO recommends reducing sodium intake.", prompt)

    def test_prompt_rejects_missing_question_or_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "Question cannot be empty"):
            build_grounded_prompt(" ", [evidence_chunk()])
        with self.assertRaisesRegex(ValueError, "evidence chunk"):
            build_grounded_prompt("Question", [])


class GrokGeneratorTests(unittest.TestCase):
    def test_generate_uses_responses_api_without_server_storage(self) -> None:
        client = FakeClient()
        generator = GrokGenerator(model="test-grok", client=client)

        answer = generator.generate("What is recommended?", [evidence_chunk()])

        self.assertEqual(answer.text, "Grounded answer [sodium_chunk_001]")
        self.assertEqual(answer.provider, "xAI")
        self.assertEqual(answer.model, "test-grok")
        self.assertEqual(answer.evidence_chunk_ids, ("sodium_chunk_001",))
        request = client.responses.calls[0]
        self.assertEqual(request["model"], "test-grok")
        self.assertIs(request["store"], False)
        self.assertEqual(request["input"][0]["role"], "system")
        self.assertEqual(request["input"][1]["role"], "user")

    def test_missing_api_key_has_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GrokConfigurationError, "XAI_API_KEY"):
                GrokGenerator()

    def test_empty_api_response_is_rejected(self) -> None:
        generator = GrokGenerator(
            client=FakeClient(FakeResponses(output_text=" ")),
        )

        with self.assertRaisesRegex(GrokGenerationError, "empty answer"):
            generator.generate("Question", [evidence_chunk()])


if __name__ == "__main__":
    unittest.main()
