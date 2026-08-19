"""Proactive multi-project scheduling for Gemini generation requests."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
from threading import Lock
import time
from typing import Any, Callable, Sequence, TypeVar

from src.generation.rate_limiter import SlidingWindowRateLimiter


try:
    from httpx import TransportError as HttpxTransportError
except ImportError:  # pragma: no cover - google-genai installs httpx in production.
    HttpxTransportError = None  # type: ignore[assignment,misc]


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class GeminiSchedulerConfig:
    """Quota and bounded fallback settings shared by all Gemini projects."""

    rpm_limit: int = 15
    safe_rpm: int = 14
    window_seconds: float = 60.0
    max_attempts: int = 6
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    backoff_jitter_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.rpm_limit < 1:
            raise ValueError("rpm_limit must be at least 1")
        if not 1 <= self.safe_rpm <= self.rpm_limit:
            raise ValueError("safe_rpm must be between 1 and rpm_limit")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.backoff_base_seconds < 0 or self.backoff_max_seconds < 0:
            raise ValueError("backoff values cannot be negative")
        if self.backoff_jitter_seconds < 0:
            raise ValueError("backoff_jitter_seconds cannot be negative")


@dataclass(frozen=True, slots=True)
class GeminiProject:
    """One independently quota-limited Gemini project client."""

    label: str
    client: Any
    limiter: SlidingWindowRateLimiter


class GeminiScheduler:
    """Round-robin scheduler with atomic per-project slot reservation."""

    def __init__(
        self,
        clients: Sequence[tuple[str, Any]],
        *,
        config: GeminiSchedulerConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
        logger: logging.Logger | None = None,
    ) -> None:
        if not clients:
            raise ValueError("At least one Gemini project client is required")
        labels = [label for label, _client in clients]
        if len(labels) != len(set(labels)):
            raise ValueError("Gemini project labels must be unique")

        self.config = config or GeminiSchedulerConfig()
        self._clock = clock
        self._sleep = sleep
        self._random_value = random_value
        self._logger = logger or logging.getLogger(__name__)
        self._projects = tuple(
            GeminiProject(
                label=label,
                client=client,
                limiter=SlidingWindowRateLimiter(
                    self.config.safe_rpm,
                    window_seconds=self.config.window_seconds,
                    clock=clock,
                ),
            )
            for label, client in clients
        )
        self._next_index = 0
        self._selection_lock = Lock()

    @property
    def project_labels(self) -> tuple[str, ...]:
        return tuple(project.label for project in self._projects)

    def _acquire_project(self) -> GeminiProject:
        while True:
            wait_seconds: float | None = None
            with self._selection_lock:
                project_count = len(self._projects)
                for offset in range(project_count):
                    index = (self._next_index + offset) % project_count
                    project = self._projects[index]
                    reservation = project.limiter.try_reserve()
                    if reservation.reserved:
                        self._next_index = (index + 1) % project_count
                        self._logger.info(
                            "%s usage: %d/%d requests in current %.0f-second window",
                            project.label,
                            reservation.usage,
                            reservation.limit,
                            self.config.window_seconds,
                        )
                        self._logger.info("Selected %s for Gemini request", project.label)
                        return project

                    self._logger.info(
                        "%s rate capacity reached or temporarily unavailable",
                        project.label,
                    )
                    if reservation.retry_after > 0:
                        wait_seconds = (
                            reservation.retry_after
                            if wait_seconds is None
                            else min(wait_seconds, reservation.retry_after)
                        )

            delay = max(wait_seconds or 0.001, 0.001)
            self._logger.info(
                "All Gemini projects are at local RPM capacity; waiting %.3f seconds",
                delay,
            )
            self._sleep(delay)

    def _backoff_seconds(self, failed_attempt: int) -> float:
        exponential = min(
            self.config.backoff_base_seconds * (2 ** max(failed_attempt - 1, 0)),
            self.config.backoff_max_seconds,
        )
        return exponential + (
            self._random_value() * self.config.backoff_jitter_seconds
        )

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        candidates = [
            getattr(error, "status_code", None),
            getattr(error, "code", None),
            getattr(getattr(error, "response", None), "status_code", None),
        ]
        for candidate in candidates:
            if candidate is None or callable(candidate):
                continue
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _is_retryable(cls, error: Exception) -> tuple[bool, int | None]:
        status_code = cls._status_code(error)
        if status_code == 429 or (status_code is not None and 500 <= status_code < 600):
            return True, status_code
        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            return True, status_code
        if HttpxTransportError is not None and isinstance(error, HttpxTransportError):
            return True, status_code
        return False, status_code

    def execute(self, operation: Callable[[Any], ResultT]) -> ResultT:
        """Run one Gemini call with proactive scheduling and bounded fallback."""

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            project = self._acquire_project()
            try:
                return operation(project.client)
            except Exception as error:
                retryable, status_code = self._is_retryable(error)
                if not retryable:
                    raise
                last_error = error
                delay = self._backoff_seconds(attempt)
                project.limiter.mark_unavailable(delay)
                failure_label = str(status_code) if status_code is not None else "network"
                self._logger.warning(
                    "%s Gemini request failed with retryable %s error on attempt %d/%d; "
                    "temporarily unavailable for %.3f seconds",
                    project.label,
                    failure_label,
                    attempt,
                    self.config.max_attempts,
                    delay,
                )

        assert last_error is not None
        raise last_error

    def usage(self) -> dict[str, tuple[int, int, float]]:
        """Return safe per-label usage snapshots for diagnostics and tests."""

        return {
            project.label: (
                snapshot.usage,
                snapshot.limit,
                snapshot.retry_after,
            )
            for project in self._projects
            for snapshot in [project.limiter.snapshot()]
        }
