"""Generate grounded answers through proactively scheduled Gemini projects."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol, Sequence

from dotenv import load_dotenv

from src.generation.common import (
    GROUNDED_RESPONSE_JSON_SCHEMA,
    GeneratedAnswer,
    SYSTEM_PROMPT,
    build_grounded_prompt,
    parse_generated_answer,
)
from src.generation.gemini_scheduler import GeminiScheduler, GeminiSchedulerConfig
from src.reranking import RerankedChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_API_TIMEOUT_SECONDS = 60.0
DEFAULT_GEMINI_RPM_LIMIT = 15
DEFAULT_GEMINI_SAFE_RPM = 14
DEFAULT_GEMINI_MAX_ATTEMPTS = 6


class InteractionsAPI(Protocol):
    """Minimal Interactions API interface needed from the Gemini client."""

    def create(self, **kwargs: Any) -> Any: ...


class GeminiClient(Protocol):
    """Minimal client interface used by :class:`GeminiGenerator`."""

    interactions: InteractionsAPI


ClientFactory = Callable[[str, float], GeminiClient]


class GeminiConfigurationError(RuntimeError):
    """Raised when Gemini is missing required configuration."""


class GeminiGenerationError(RuntimeError):
    """Raised when Gemini cannot return a usable answer."""


_SHARED_SCHEDULER_LOCK = Lock()
_SHARED_SCHEDULER: GeminiScheduler | None = None
_SHARED_SCHEDULER_SIGNATURE: tuple[Any, ...] | None = None


def _integer_environment(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise GeminiConfigurationError(f"{name} must be an integer") from exc


def _float_environment(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        return float(raw_value)
    except ValueError as exc:
        raise GeminiConfigurationError(f"{name} must be numeric") from exc


def _scheduler_config_from_environment() -> GeminiSchedulerConfig:
    try:
        return GeminiSchedulerConfig(
            rpm_limit=_integer_environment(
                "GEMINI_RPM_LIMIT",
                DEFAULT_GEMINI_RPM_LIMIT,
            ),
            safe_rpm=_integer_environment(
                "GEMINI_SAFE_RPM",
                DEFAULT_GEMINI_SAFE_RPM,
            ),
            max_attempts=_integer_environment(
                "GEMINI_MAX_ATTEMPTS",
                DEFAULT_GEMINI_MAX_ATTEMPTS,
            ),
            backoff_base_seconds=_float_environment(
                "GEMINI_BACKOFF_BASE_SECONDS",
                0.5,
            ),
            backoff_max_seconds=_float_environment(
                "GEMINI_BACKOFF_MAX_SECONDS",
                8.0,
            ),
            backoff_jitter_seconds=_float_environment(
                "GEMINI_BACKOFF_JITTER_SECONDS",
                0.25,
            ),
        )
    except ValueError as exc:
        raise GeminiConfigurationError(f"Invalid Gemini scheduler configuration: {exc}") from exc


def _configured_api_keys(api_key: str | None) -> list[tuple[str, str]]:
    if api_key:
        return [("key_1", api_key)]

    configured = [
        (f"key_{index}", value)
        for index in range(1, 4)
        if (value := os.getenv(f"GEMINI_API_KEY_{index}"))
    ]
    if not configured:
        legacy_key = os.getenv("GEMINI_API_KEY")
        if legacy_key:
            configured = [("key_1", legacy_key)]
    if not configured:
        raise GeminiConfigurationError(
            "No Gemini API keys are configured. Set GEMINI_API_KEY_1, "
            "GEMINI_API_KEY_2, and GEMINI_API_KEY_3 in .env."
        )

    raw_keys = [key for _label, key in configured]
    if len(raw_keys) != len(set(raw_keys)):
        raise GeminiConfigurationError(
            "Gemini API key environment variables must contain distinct project keys."
        )
    return configured


def _default_client_factory(api_key: str, timeout_seconds: float) -> GeminiClient:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise GeminiConfigurationError(
            "google-genai is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
    )


def _scheduler_signature(
    configured_keys: Sequence[tuple[str, str]],
    config: GeminiSchedulerConfig,
    timeout_seconds: float,
) -> tuple[Any, ...]:
    key_fingerprints = tuple(
        (label, hashlib.sha256(api_key.encode("utf-8")).digest())
        for label, api_key in configured_keys
    )
    return key_fingerprints, config, timeout_seconds


def _shared_scheduler(
    configured_keys: Sequence[tuple[str, str]],
    config: GeminiSchedulerConfig,
    timeout_seconds: float,
) -> GeminiScheduler:
    global _SHARED_SCHEDULER, _SHARED_SCHEDULER_SIGNATURE

    signature = _scheduler_signature(configured_keys, config, timeout_seconds)
    with _SHARED_SCHEDULER_LOCK:
        if _SHARED_SCHEDULER is None or _SHARED_SCHEDULER_SIGNATURE != signature:
            clients = [
                (label, _default_client_factory(api_key, timeout_seconds))
                for label, api_key in configured_keys
            ]
            _SHARED_SCHEDULER = GeminiScheduler(clients, config=config)
            _SHARED_SCHEDULER_SIGNATURE = signature
        return _SHARED_SCHEDULER


class GeminiGenerator:
    """Call Gemini while keeping project selection internal to generation."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_API_TIMEOUT_SECONDS,
        client: GeminiClient | None = None,
        scheduler: GeminiScheduler | None = None,
        client_factory: ClientFactory | None = None,
        scheduler_config: GeminiSchedulerConfig | None = None,
    ) -> None:
        cleaned_model = model.strip()
        if not cleaned_model:
            raise ValueError("Gemini model name cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if client is not None and scheduler is not None:
            raise ValueError("Provide either client or scheduler, not both")

        self.model = cleaned_model
        self.client = client
        self.scheduler = scheduler
        if client is not None or scheduler is not None:
            return

        configured_keys = _configured_api_keys(api_key)
        config = scheduler_config or _scheduler_config_from_environment()
        if client_factory is not None:
            clients = [
                (label, client_factory(key, timeout_seconds))
                for label, key in configured_keys
            ]
            self.scheduler = GeminiScheduler(clients, config=config)
        elif api_key is not None:
            clients = [
                (label, _default_client_factory(key, timeout_seconds))
                for label, key in configured_keys
            ]
            self.scheduler = GeminiScheduler(clients, config=config)
        else:
            self.scheduler = _shared_scheduler(
                configured_keys,
                config,
                timeout_seconds,
            )

    def generate(
        self,
        question: str,
        evidence: Sequence[RerankedChunk],
    ) -> GeneratedAnswer:
        """Generate one answer grounded in the supplied reranked evidence."""

        prompt = build_grounded_prompt(question, evidence)

        def create_interaction(client: GeminiClient) -> Any:
            return client.interactions.create(
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

        try:
            if self.scheduler is not None:
                interaction = self.scheduler.execute(create_interaction)
            else:
                assert self.client is not None
                interaction = create_interaction(self.client)
        except Exception as exc:
            raise GeminiGenerationError(
                "Gemini API request failed after bounded retry handling"
            ) from exc

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
