"""Grounded LLM generation for food-suitability assessments.

Calls the Anthropic API with the retrieved evidence chunks and parses the
response into a `FoodAssessment`. Requires the `anthropic` package
(added to requirements.txt) and an `ANTHROPIC_API_KEY` environment variable.

If you'd rather use a different provider (OpenAI, local model, etc.), only
this file needs to change -- `prompts.py` and `schemas.py` are provider
agnostic.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

from anthropic import Anthropic

from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.generation.schemas import Citation, FoodAssessment

try:
    from src.retrieval import HybridRetrievedChunk
except ImportError:  # pragma: no cover
    HybridRetrievedChunk = object  # type: ignore[misc,assignment]


# Check Anthropic's current model list before relying on this string long-term;
# model names get deprecated/replaced over time.
DEFAULT_MODEL = "claude-sonnet-4-6"


class GenerationError(RuntimeError):
    """Raised when the LLM response can't be parsed into a FoodAssessment."""


class FoodAssessmentGenerator:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model
        self._max_tokens = max_tokens

    def generate(
        self,
        question: str,
        chunks: Sequence["HybridRetrievedChunk"],
        food_name_hint: str | None = None,
    ) -> FoodAssessment:
        """Generate a grounded FoodAssessment from retrieved evidence.

        `food_name_hint` is optional -- if you've already extracted the food
        name upstream (e.g. via a lightweight NER/regex step), pass it in so
        the model doesn't have to re-derive it from the question.
        """

        user_prompt = build_user_prompt(question, chunks)
        if food_name_hint:
            user_prompt += f"\n\n(Hint: the food in question is '{food_name_hint}'.)"

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return self._parse(raw_text, chunks)

    @staticmethod
    def _parse(raw_text: str, chunks: Sequence["HybridRetrievedChunk"]) -> FoodAssessment:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GenerationError(f"Model did not return valid JSON: {raw_text!r}") from exc

        # Cross-check citations against chunk_ids that were actually retrieved,
        # to catch the model hallucinating a chunk_id that was never provided.
        known_chunk_ids = {chunk.chunk_id for chunk in chunks}
        citations = []
        for c in data.get("citations", []):
            chunk_id = c.get("chunk_id", "")
            if chunk_id not in known_chunk_ids:
                # Skip rather than trust an unverifiable citation.
                continue
            citations.append(
                Citation(
                    document_name=c.get("document_name", "Unknown"),
                    chunk_id=chunk_id,
                    section_title=c.get("section_title"),
                    page_start=c.get("page_start"),
                    page_end=c.get("page_end"),
                )
            )

        try:
            return FoodAssessment(
                food_name=data["food_name"],
                assessment=data["assessment"],
                reasoning=data["reasoning"],
                recommendations=data.get("recommendations", []),
                supporting_evidence=data.get("supporting_evidence", ""),
                citations=citations,
                confidence=data.get("confidence", "Low"),
            )
        except (KeyError, ValueError) as exc:
            raise GenerationError(f"Model JSON missing/invalid fields: {data!r}") from exc
