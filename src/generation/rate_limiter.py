"""Thread-safe sliding-window request limiter used by hosted model projects."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
import time
from typing import Callable


@dataclass(frozen=True, slots=True)
class Reservation:
    """Result of an atomic capacity check and optional reservation."""

    reserved: bool
    usage: int
    limit: int
    retry_after: float


class SlidingWindowRateLimiter:
    """Track one project's requests in an independent rolling time window."""

    def __init__(
        self,
        max_requests: int,
        *,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._timestamps: deque[float] = deque()
        self._blocked_until = 0.0
        self._lock = Lock()

    def _expire(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def try_reserve(self) -> Reservation:
        """Atomically reserve one request when local capacity is available."""

        now = self._clock()
        with self._lock:
            self._expire(now)
            if now < self._blocked_until:
                return Reservation(
                    reserved=False,
                    usage=len(self._timestamps),
                    limit=self.max_requests,
                    retry_after=self._blocked_until - now,
                )
            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return Reservation(
                    reserved=True,
                    usage=len(self._timestamps),
                    limit=self.max_requests,
                    retry_after=0.0,
                )
            retry_after = max(
                0.0,
                self._timestamps[0] + self.window_seconds - now,
            )
            return Reservation(
                reserved=False,
                usage=len(self._timestamps),
                limit=self.max_requests,
                retry_after=retry_after,
            )

    def mark_unavailable(self, seconds: float) -> None:
        """Temporarily remove this project from scheduling after an API failure."""

        if seconds < 0:
            raise ValueError("seconds cannot be negative")
        now = self._clock()
        with self._lock:
            self._blocked_until = max(self._blocked_until, now + seconds)

    def snapshot(self) -> Reservation:
        """Return current usage and time until capacity without reserving."""

        now = self._clock()
        with self._lock:
            self._expire(now)
            if now < self._blocked_until:
                retry_after = self._blocked_until - now
            elif len(self._timestamps) >= self.max_requests:
                retry_after = max(
                    0.0,
                    self._timestamps[0] + self.window_seconds - now,
                )
            else:
                retry_after = 0.0
            return Reservation(
                reserved=False,
                usage=len(self._timestamps),
                limit=self.max_requests,
                retry_after=retry_after,
            )
