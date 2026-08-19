"""Generate evidence-grounded answers with Google's hosted Gemini API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, Sequence

from dotenv import load_dotenv

from src.generation.common import (
    GROUNDED_RESPONSE_JSON_SCHEMA,
    GeneratedAnswer,
    SYSTEM_PROMPT,
    build_grounded_prompt,
    parse_generated_answer,
)
from src.reranking import RerankedChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_API_TIMEOUT_SECONDS = 60.0


class InteractionsAPI(Protocol):
    """Minimal Interactions API interface needed from the Gemini client."""

    def create(self, **kwargs: Any) -> Any: ...


class GeminiClient(Protocol):
    """Minimal client interface used by :class:`GeminiGenerator`."""

    interactions: InteractionsAPI


class GeminiConfigurationError(RuntimeError):
    """Raised when Gemini is missing required configuration."""


class GeminiGenerationError(RuntimeError):
    """Raised when Gemini cannot return a usable answer."""


class GeminiGenerator:
    """Call Gemini through Google's stateless Interactions API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_API_TIMEOUT_SECONDS,
        client: GeminiClient | None = None,
    ) -> None:
        cleaned_model = model.strip()
        if not cleaned_model:
            raise ValueError("Gemini model name cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self.model = cleaned_model
        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is not set. Add the Gemini API key to .env "
                "before running generation."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiConfigurationError(
                "google-genai is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc
        self.client = genai.Client(
            api_key=resolved_api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    def generate(
        self,
        question: str,
        evidence: Sequence[RerankedChunk],
    ) -> GeneratedAnswer:
        """Generate one answer grounded in the supplied reranked evidence."""

        prompt = build_grounded_prompt(question, evidence)
        try:
            interaction = self.client.interactions.create(
                model=self.model,
                system_instruction=SYSTEM_PROMPT,
                input=prompt,
                store=False,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": GROUNDED_RESPONSE_JSON_SCHEMA,
                },
            )
        except Exception as exc:
            raise GeminiGenerationError(f"Gemini API request failed: {exc}") from exc

        answer_text = str(getattr(interaction, "output_text", "") or "").strip()
        if not answer_text:
            raise GeminiGenerationError("Gemini API returned an empty answer")

        try:
            return parse_generated_answer(
                answer_text,
                evidence,
                provider="Google Gemini",
                model=self.model,
            )
        except ValueError as exc:
            raise GeminiGenerationError(
                f"Gemini returned an invalid grounded answer: {exc}"
            ) from exc
