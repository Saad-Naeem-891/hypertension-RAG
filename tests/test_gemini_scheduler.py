import io
import logging
import os
from threading import Barrier, Lock, Thread
import unittest
from unittest.mock import patch

import httpx

from src.generation.gemini_generator import GeminiGenerator
from src.generation.gemini_scheduler import GeminiScheduler, GeminiSchedulerConfig
from src.generation.rate_limiter import SlidingWindowRateLimiter


class FakeClock:
    """Controllable monotonic clock that makes limiter tests instantaneous."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self._lock = Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


class RetryableHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str = "request failed") -> None:
        super().__init__(message)
        self.status_code = status_code


def scheduler_config(**overrides) -> GeminiSchedulerConfig:
    values = {
        "rpm_limit": 15,
        "safe_rpm": 14,
        "window_seconds": 60.0,
        "max_attempts": 6,
        "backoff_base_seconds": 1.0,
        "backoff_max_seconds": 8.0,
        "backoff_jitter_seconds": 0.0,
    }
    values.update(overrides)
    return GeminiSchedulerConfig(**values)


class GeminiSchedulerTests(unittest.TestCase):
    def make_scheduler(
        self,
        labels=("key_1", "key_2", "key_3"),
        *,
        clock: FakeClock | None = None,
        **config_overrides,
    ) -> tuple[GeminiScheduler, FakeClock]:
        fake_clock = clock or FakeClock()
        scheduler = GeminiScheduler(
            [(label, label) for label in labels],
            config=scheduler_config(**config_overrides),
            clock=fake_clock.monotonic,
            sleep=fake_clock.sleep,
            random_value=lambda: 0.0,
        )
        return scheduler, fake_clock

    def test_three_projects_are_selected_in_round_robin_order(self) -> None:
        scheduler, _clock = self.make_scheduler(safe_rpm=2)

        selected = [scheduler.execute(lambda client: client) for _ in range(6)]

        self.assertEqual(
            selected,
            ["key_1", "key_2", "key_3", "key_1", "key_2", "key_3"],
        )

    def test_full_project_is_skipped_when_another_has_capacity(self) -> None:
        scheduler, clock = self.make_scheduler(
            labels=("key_1", "key_2"),
            safe_rpm=1,
        )

        first = scheduler.execute(lambda client: client)
        second = scheduler.execute(lambda client: client)

        self.assertEqual((first, second), ("key_1", "key_2"))
        self.assertEqual(clock.sleeps, [])

    def test_all_full_projects_wait_for_earliest_available_slot(self) -> None:
        scheduler, clock = self.make_scheduler(
            labels=("key_1", "key_2"),
            safe_rpm=1,
        )
        scheduler.execute(lambda client: client)
        scheduler.execute(lambda client: client)

        selected = scheduler.execute(lambda client: client)

        self.assertEqual(selected, "key_1")
        self.assertEqual(clock.sleeps, [60.0])

    def test_request_expires_after_rolling_window(self) -> None:
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(
            1,
            window_seconds=60.0,
            clock=clock.monotonic,
        )

        self.assertTrue(limiter.try_reserve().reserved)
        self.assertFalse(limiter.try_reserve().reserved)
        clock.advance(60.0)

        reservation = limiter.try_reserve()
        self.assertTrue(reservation.reserved)
        self.assertEqual(reservation.usage, 1)

    def test_project_windows_are_independent(self) -> None:
        scheduler, _clock = self.make_scheduler(
            labels=("key_1", "key_2"),
            safe_rpm=2,
        )

        scheduler.execute(lambda client: client)
        scheduler.execute(lambda client: client)
        scheduler.execute(lambda client: client)

        usage = scheduler.usage()
        self.assertEqual(usage["key_1"][:2], (2, 2))
        self.assertEqual(usage["key_2"][:2], (1, 2))

    def test_concurrent_reservations_do_not_exceed_safe_rpm(self) -> None:
        scheduler, _clock = self.make_scheduler(safe_rpm=2)
        barrier = Barrier(6)
        selected: list[str] = []
        selected_lock = Lock()

        def worker() -> None:
            barrier.wait()
            project = scheduler.execute(lambda client: client)
            with selected_lock:
                selected.append(project)

        threads = [Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(selected), 6)
        self.assertEqual(
            {label: value[:2] for label, value in scheduler.usage().items()},
            {"key_1": (2, 2), "key_2": (2, 2), "key_3": (2, 2)},
        )

    def test_429_fails_over_to_another_project(self) -> None:
        scheduler, clock = self.make_scheduler()
        called: list[str] = []

        def operation(client: str) -> str:
            called.append(client)
            if client == "key_1":
                raise RetryableHTTPError(429)
            return client

        result = scheduler.execute(operation)

        self.assertEqual(result, "key_2")
        self.assertEqual(called, ["key_1", "key_2"])
        self.assertEqual(clock.sleeps, [])

    def test_5xx_and_network_errors_are_retryable(self) -> None:
        scheduler, _clock = self.make_scheduler()
        errors = [RetryableHTTPError(503), TimeoutError("timeout")]

        def operation(client: str) -> str:
            if errors:
                raise errors.pop(0)
            return client

        self.assertEqual(scheduler.execute(operation), "key_3")

    def test_httpx_transport_errors_used_by_gemini_sdk_are_retryable(self) -> None:
        scheduler, _clock = self.make_scheduler(labels=("key_1", "key_2"))
        attempts = 0

        def operation(client: str) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                request = httpx.Request("POST", "https://example.invalid")
                raise httpx.ConnectError("connection failed", request=request)
            return client

        self.assertEqual(scheduler.execute(operation), "key_2")

    def test_non_retryable_error_is_returned_immediately(self) -> None:
        scheduler, _clock = self.make_scheduler()
        calls = 0

        def operation(_client: str) -> str:
            nonlocal calls
            calls += 1
            raise RetryableHTTPError(400)

        with self.assertRaises(RetryableHTTPError):
            scheduler.execute(operation)
        self.assertEqual(calls, 1)

    def test_bounded_retries_use_exponential_backoff(self) -> None:
        scheduler, clock = self.make_scheduler(
            labels=("key_1",),
            max_attempts=3,
            backoff_base_seconds=1.0,
            backoff_max_seconds=8.0,
        )
        calls = 0

        def operation(_client: str) -> str:
            nonlocal calls
            calls += 1
            raise RetryableHTTPError(429)

        with self.assertRaises(RetryableHTTPError):
            scheduler.execute(operation)

        self.assertEqual(calls, 3)
        self.assertEqual(clock.sleeps, [1.0, 2.0])

    def test_configurable_safe_rpm_is_enforced(self) -> None:
        scheduler, _clock = self.make_scheduler(
            labels=("key_1",),
            rpm_limit=4,
            safe_rpm=3,
        )
        for _ in range(3):
            scheduler.execute(lambda client: client)

        self.assertEqual(scheduler.usage()["key_1"][:2], (3, 3))

    def test_one_or_two_configured_keys_are_supported(self) -> None:
        for labels in (("key_1",), ("key_1", "key_2")):
            scheduler, _clock = self.make_scheduler(labels=labels)
            self.assertEqual(scheduler.project_labels, labels)
            self.assertIn(scheduler.execute(lambda client: client), labels)

    def test_logs_use_labels_and_never_expose_api_keys(self) -> None:
        secret = "super-secret-api-key"
        stream = io.StringIO()
        logger = logging.getLogger("gemini-scheduler-secret-test")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(stream))
        clock = FakeClock()
        scheduler = GeminiScheduler(
            [("key_1", object()), ("key_2", object())],
            config=scheduler_config(),
            clock=clock.monotonic,
            sleep=clock.sleep,
            random_value=lambda: 0.0,
            logger=logger,
        )
        attempts = 0

        def operation(_client: object) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RetryableHTTPError(429, f"failed for {secret}")
            return "ok"

        self.assertEqual(scheduler.execute(operation), "ok")
        output = stream.getvalue()
        self.assertIn("key_1", output)
        self.assertIn("key_2", output)
        self.assertNotIn(secret, output)

    def test_generator_reads_numbered_keys_and_quota_configuration(self) -> None:
        environment = {
            "GEMINI_API_KEY_1": "secret-one",
            "GEMINI_API_KEY_2": "secret-two",
            "GEMINI_RPM_LIMIT": "12",
            "GEMINI_SAFE_RPM": "11",
        }
        created_keys: list[str] = []

        def client_factory(api_key: str, _timeout: float) -> object:
            created_keys.append(api_key)
            return object()

        with patch.dict(os.environ, environment, clear=True):
            generator = GeminiGenerator(client_factory=client_factory)

        self.assertEqual(created_keys, ["secret-one", "secret-two"])
        self.assertIsNotNone(generator.scheduler)
        assert generator.scheduler is not None
        self.assertEqual(generator.scheduler.project_labels, ("key_1", "key_2"))
        self.assertEqual(generator.scheduler.config.rpm_limit, 12)
        self.assertEqual(generator.scheduler.config.safe_rpm, 11)


if __name__ == "__main__":
    unittest.main()
