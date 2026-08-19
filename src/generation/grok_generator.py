"""Generate evidence-grounded answers with the hosted xAI Grok API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, Sequence

from dotenv import load_dotenv

from src.generation.common import (
    GeneratedAnswer,
    SYSTEM_PROMPT,
    build_grounded_prompt,
    parse_generated_answer,
)
from src.reranking import RerankedChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_GROK_MODEL = "grok-4.20-non-reasoning"
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_API_TIMEOUT_SECONDS = 60.0


class ResponsesAPI(Protocol):
    """Minimal Responses API interface needed from the OpenAI-compatible client."""

    def create(self, **kwargs: Any) -> Any: ...


class GrokClient(Protocol):
    """Minimal client interface used by :class:`GrokGenerator`."""

    responses: ResponsesAPI


class GrokConfigurationError(RuntimeError):
    """Raised when the Grok client is missing required configuration."""


class GrokGenerationError(RuntimeError):
    """Raised when Grok cannot return a usable answer."""


class GrokGenerator:
    """Call Grok through xAI's OpenAI-compatible Responses API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GROK_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_XAI_BASE_URL,
        timeout_seconds: float = DEFAULT_API_TIMEOUT_SECONDS,
        client: GrokClient | None = None,
    ) -> None:
        cleaned_model = model.strip()
        if not cleaned_model:
            raise ValueError("Grok model name cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self.model = cleaned_model
        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key or os.getenv("XAI_API_KEY")
        if not resolved_api_key:
            raise GrokConfigurationError(
                "XAI_API_KEY is not set. Add the xAI API key to .env before "
                "running Grok generation."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GrokConfigurationError(
                "openai is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    def generate(
        self,
        question: str,
        evidence: Sequence[RerankedChunk],
    ) -> GeneratedAnswer:
        """Generate one answer grounded in the supplied reranked evidence."""

        prompt = build_grounded_prompt(question, evidence)
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                store=False,
            )
        except Exception as exc:
            raise GrokGenerationError(f"Grok API request failed: {exc}") from exc

        answer_text = str(getattr(response, "output_text", "") or "").strip()
        if not answer_text:
            raise GrokGenerationError("Grok API returned an empty answer")

        try:
            return parse_generated_answer(
                answer_text,
                evidence,
                provider="xAI",
                model=self.model,
            )
        except ValueError as exc:
            raise GrokGenerationError(
                f"Grok returned an invalid grounded answer: {exc}"
            ) from exc
